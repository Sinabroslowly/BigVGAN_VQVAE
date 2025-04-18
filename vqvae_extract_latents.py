import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import VQModel
from scripts.dataset_vqvae import AudioDataset

def extract_latents(model, data_loader, device, output_dir, accelerator):
    """
    Extract latents from the VQ-VAE model and save them with the same name as input files but with a .pt extension.
    Latents are saved separately for train, val, and test phases.

    Args:
        model: The VQ-VAE model.
        data_loader: DataLoader for the current phase.
        device: Device (CPU or GPU).
        output_dir: Directory to save extracted latents.
        accelerator: Accelerate object for distributed support.
    """
    model.eval()  # Set model to evaluation mode
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (audio_spec, audio_paths) in enumerate(tqdm(data_loader, desc=f"Extracting latents")):
            # Move data to device
            input_data = audio_spec.to(device)

            # Extract intermediate latent outputs
            #latents = model.module.encode(input_data).latents  # Get latents (before quantization)
            raw_latents = model.encode(input_data).latents  # Get latents (before quantization)
            # latents = model.module.quant_conv(latents)  # Apply quantization convolution
            # quantized_latents, _, _ = model.module.quantize(latents)  # Quantized latents

            # Save each latent
            for i, audio_path in enumerate(audio_paths):
                # Replace the file extension with .pt
                audio_name = os.path.splitext(os.path.basename(audio_path))[0]
                latent_path = os.path.join(output_dir, f"{audio_name}.pt")
                torch.save(raw_latents[i].cpu(), latent_path)

            # Log progress
            accelerator.print(f"Batch {batch_idx + 1}/{len(data_loader)}: Latents saved to {output_dir} with dimension {raw_latents.shape}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="./datasets_subset", help="Path to dataset")
    # parser.add_argument("--output_dir", type=str, default="/home/airis_lab/MJ/RIRLDM/datasets_mel_subset_complete", help="Root directory to save extracted latents")
    parser.add_argument("--dataset", type=str, default="./datasets", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="D:\Study\KAIST\MS_thesis/rir_ldm_local\datasets", help="Root directory to save extracted latents")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for data loader")
    parser.add_argument("--from_pretrained", type=str, default="epoch_240_checkpoint.pth", help="Path to pretrained VQ-VAE model checkpoint")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints_vqvae", help="The path to global checkpoint")
    parser.add_argument("--num_embeddings", type=int, default=512, help="Number of embeddings for VQ-VAE.")
    parser.add_argument("--version", type=str, default="vqvae_01", help="The checkpoint version to extract")
    args = parser.parse_args()

    # Setup accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Load model
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

    checkpoint = torch.load(os.path.join(args.checkpoints, args.version, args.from_pretrained), map_location=device)
    state_dict = checkpoint["model_state"]
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Total {sum(p.numel() for p in model.parameters())} parameters available in model.")
    model = accelerator.prepare(model)

    # Phases to process
    phases = ["train", "val", "test"]

    for phase in phases:
        # Prepare DataLoader for each phase
        dataset = AudioDataset(dataroot=args.dataset, phase=phase)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)

        # Output directory for the current phase
        phase_output_dir = os.path.join(args.output_dir, phase + "_L")

        # Run latent extraction
        accelerator.print(f"Processing {phase} phase. Saving to {phase_output_dir}")
        extract_latents(model, data_loader, device, phase_output_dir, accelerator)

if __name__ == "__main__":
    main()
