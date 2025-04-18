import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
import pyroomacoustics
from diffusers import VQModel
from accelerate import Accelerator
import argparse
# import lpips
import numpy as np
import matplotlib.pyplot as plt

from scripts_vqvae.dataset_vqvae import AudioDataset
from scripts_vqvae.meldataset import get_mel_spectrogram



LR = 1e-4
ADAM_BETA = (0.8, 0.99)
ADAM_EPS = 1e-8
LAMBDAS = [1, 1e-2, 1, 1] # [Spec Recon, Quantization, T60_error, LPIPS Loss]

def train(model, train_loader, val_loader, optimizer, scheduler, device, start_epoch, best_val_loss, args, accelerator):
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.logs, args.version))
        colormap = plt.get_cmap("inferno")
    
    for epoch in range(start_epoch + 1, args.epochs):
        model.train()
        train_loss_total = 0
        train_loss_1 = 0
        train_loss_2 = 0
        # train_loss_3 = 0

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train: Epoch {epoch}/{args.epochs}")
        for _ , batch in train_pbar:
            audio_spec, _ = batch
            input_data = audio_spec.to(device)
            output, commitment_loss = model(input_data, return_dict=False)
            # Loss 1: Reconstruction loss MAE
            reconstruction_spec_loss = nn.functional.l1_loss(output, input_data)
            
            # Loss 3: RT60 loss (based on pyroomacoustics)
            # y_r = [stft.inverse(s.squeeze().clone()) for s in input_data]
            # y_f = [stft.inverse(s.squeeze().clone()) for s in output]

            # t60_loss = 1
            # try:
            #     f = lambda x: pyroomacoustics.experimental.rt60.measure_rt60(x, 22050)
            #     t60_r = [f(y) for y in y_r if len(y)]
            #     t60_f = [f(y) for y in y_f if len(y)]
            #     t60_loss = np.mean([((t_b - t_a) / t_a) for t_a, t_b in zip(t60_r, t60_f)])
            # except:
            #     pass

            loss = (
                LAMBDAS[0] * reconstruction_spec_loss +
                LAMBDAS[1] * args.commitment_cost * commitment_loss# +
                # LAMBDAS[2] * t60_loss
            )

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            train_loss_total += loss.item()
            train_loss_1 += reconstruction_spec_loss.item()
            train_loss_2 += commitment_loss.item()
            # train_loss_3 += t60_loss

        # Logging training metrics
        if accelerator.is_main_process:
            writer.add_scalar("Train/Total_Train", train_loss_total / len(train_loader), epoch)
            writer.add_scalar("Train/Reconstruction_Train", train_loss_1 / len(train_loader), epoch)
            writer.add_scalar("Train/Commitment_Train", train_loss_2 / len(train_loader), epoch)
            # writer.add_scalar("Train/T60_Train", train_loss_3 / len(train_loader), epoch)

            print(f"Epoch {epoch}/{args.epochs}, Loss: {train_loss_total / len(train_loader):.4f}")

        # Validation Step
        model.eval()
        val_loss_total = 0
        val_loss_1 = 0
        val_loss_2 = 0
        # val_loss_3 = 0
        # val_loss_4 = 0

        with torch.no_grad():
            val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation: Epoch {epoch}/{args.epochs}")
            for _, batch in val_pbar:
                audio_spec, _ = batch
                input_data = audio_spec.to(device)
                output, commitment_loss = model(input_data, return_dict=False)

                reconstruction_spec_loss = nn.functional.l1_loss(output, input_data)
                # y_r = [stft.inverse(s.squeeze().clone()) for s in input_data]
                # y_f = [stft.inverse(s.squeeze().clone()) for s in output]

                # t60_loss = 1
                # try:
                #     f = lambda x: pyroomacoustics.experimental.rt60.measure_rt60(x, 22050)
                #     t60_r = [f(y) for y in y_r if len(y)]
                #     t60_f = [f(y) for y in y_f if len(y)]
                #     t60_loss = np.mean([((t_b - t_a) / t_a) for t_a, t_b in zip(t60_r, t60_f)])
                # except:
                #     pass

                original_spec_rgb = input_data.repeat(1, 3, 1, 1) # [batch, 3, height, width]
                reconstructed_spec_rgb = output.repeat(1, 3, 1, 1) # [batch, 3, height, width]
                # lpips_spec_loss = lpips_loss(original_spec_rgb, reconstructed_spec_rgb).mean()

                loss = (
                    LAMBDAS[0] * reconstruction_spec_loss +
                    LAMBDAS[1] * args.commitment_cost * commitment_loss# +
                    # LAMBDAS[2] * t60_loss# +
                    # LAMBDAS[3] * lpips_spec_loss
                )

                val_loss_total += loss.item()
                val_loss_1 += reconstruction_spec_loss.item()
                val_loss_2 += commitment_loss.item()
                # val_loss_3 += t60_loss
                # val_loss_4 += lpips_spec_loss

            gen_latent = model.encode(input_data).latents
            print(f"Shape of gen_latent = {gen_latent[0].shape}")

        # Logging validation metrics
        if accelerator.is_main_process:
            input_grid = make_grid(input_data.cpu(), normalize=True)[0].squeeze()
            output_grid = make_grid(output.cpu(), normalize=True)[0].squeeze()
            latent_grid = make_grid(gen_latent[0].unsqueeze(1).cpu(), nrow=8, normalize=True, scale_each=True)[0].squeeze()
            
            input_cmap = torch.tensor(colormap(input_grid)[:, :, :3]).permute(2, 0, 1)
            output_cmap = torch.tensor(colormap(output_grid)[:, :, :3]).permute(2, 0, 1)
            latent_cmap = torch.tensor(colormap(latent_grid)[:, :, :3]).permute(2, 0, 1)
            # print(f"Shape of input_grid: {input_grid.shape}")
            # print(f"Shape of output_grid: {output_grid.shape}")
            # print(f"Shape of latent_grid: {latent_grid.shape}")
            writer.add_scalar("Validation/Total_Validation", val_loss_total / len(val_loader), epoch)
            writer.add_scalar("Validation/Reconstruction_Validation", val_loss_1 / len(val_loader), epoch)
            writer.add_scalar("Validation/Commitment_Validation", val_loss_2 / len(val_loader), epoch)
            # writer.add_scalar("Validation/T60_Validation", val_loss_3 / len(val_loader), epoch)
            # writer.add_scalar("Validation/LPIPS", val_loss_4 / len(val_loader), epoch)
            writer.add_scalar("Validation/Latent Max Value", torch.max(gen_latent).cpu(), epoch)
            writer.add_scalar("Validation/Latent Min Value", torch.min(gen_latent).cpu(), epoch)
            writer.add_image("Spectrogram/Ground_Truth (Real)", input_cmap, epoch)
            writer.add_image("Spectrogram/Generated (Real)", output_cmap, epoch)
            writer.add_image("Spectrogram/Latent Representation", latent_cmap, epoch)
        
            #print(f"Validation Loss: {val_loss_total / len(val_loader):.4f}")

        # Save checkpoints
        if accelerator.is_main_process:
            if (epoch + 1) % (args.epochs // 10) == 0 or val_loss_total < best_val_loss:
                checkpoint_path = os.path.join(
                    args.checkpoints_dir,
                    args.version,
                    f"{'best_val_checkpoint' if val_loss_total < best_val_loss else f'epoch_{epoch + 1}_checkpoint'}.pth"
                )
                torch.save({
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_loss": min(val_loss_total, best_val_loss),
                    }, checkpoint_path)
                
                if val_loss_total < best_val_loss:
                    best_val_loss = val_loss_total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints_vqvae", help="Path to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("--dataset", type=str, default="./datasets", help="Dataset path")
    # parser.add_argument("--dataset", type=str, default="./datasets_subset", help="Dataset path")
    parser.add_argument("--epochs", type=int, default=800, help="Total epochs to train the model.")
    # parser.add_argument("--from_pretrained", type=str, default="epoch_80_checkpoint.pth", help="The checkpoint name for pretraining")
    parser.add_argument("--from_pretrained", type=str, default=None, help="The checkpoint name for pretraining")
    parser.add_argument("--version", type=str, default="vqvae_01", help="The version of VQ-VAE training")
    parser.add_argument("--embedding_dim", type=int, default=3, help="Embedding dimension for VQ-VAE.")
    parser.add_argument("--num_embeddings", type=int, default=512, help="Number of embeddings for VQ-VAE.")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="Commitment cost for VQ-VAE.")
    parser.add_argument("--logs", type=str, default="./logs_vqvae", help="Logging directory for Tensorboard.")
    parser.add_argument("--function_test", type=bool, default=False, help="Flag to determine model function and memory allocation test.")
    args = parser.parse_args()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoints_dir, args.version), exist_ok=True)

    # Accelerator setup
    accelerator = Accelerator()
    device = accelerator.device

    # Model setup
    model = VQModel(
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D", 
                          "DownEncoderBlock2D", 
                          "DownEncoderBlock2D"
                          ),
        up_block_types=("UpDecoderBlock2D", 
                        "UpDecoderBlock2D", 
                        "UpDecoderBlock2D"
                        ),
        block_out_channels=(32, 64, 128),
        layers_per_block=3,
        act_fn="silu",
        sample_size=(80, 512),
        latent_channels=8,
        num_vq_embeddings=args.num_embeddings,
        scaling_factor=0.18215,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=ADAM_BETA, eps=ADAM_EPS)
    print(f"total {sum(p.numel() for p in model.parameters())} trainable parameters in model.")
    start_epoch = 0
    best_val_loss = float("inf")
    if args.from_pretrained:
        checkpoint = torch.load(os.path.join(args.checkpoints_dir, args.version, args.from_pretrained), map_location=device)
        model_state_dict = checkpoint["model_state"]
        #opti_state_dict = checkpoint["optimizer_state"]
        if any(key.startswith("module.") for key in model_state_dict.keys()):
            model_state_dict = {key[len("module."):]: value for key, value in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        #print(f"Loaded checkpoint from: {args.from_pretrained}")


    train_dataset = AudioDataset(dataroot=args.dataset, phase="train")
    val_dataset = AudioDataset(dataroot=args.dataset, phase="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # lpips_loss = lpips.LPIPS(net='vgg').to(device)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
        train_loader,
        val_loader
    )
    train(model, train_loader, val_loader, optimizer, scheduler, device, start_epoch, best_val_loss, args=args, accelerator=accelerator)

if __name__ == "__main__":
    main()
