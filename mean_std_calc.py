import os
import numpy as np
import soundfile
import torch
from tqdm import tqdm
from scripts.meldataset import mel_spectrogram  # assuming this matches your project structure

AUDIO_EXTENSIONS = [".wav", ".WAV", ".aif", ".aiff", ".AIF", ".AIFF"]

def is_audio_file(filename):
    return any(filename.endswith(ext) for ext in AUDIO_EXTENSIONS)

def collect_audio_paths(root_dir):
    audio_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if is_audio_file(file):
                audio_paths.append(os.path.join(root, file))
    return audio_paths

def compute_mean_std(audio_dir, sr=22050):
    all_logmel_values = []

    audio_paths = collect_audio_paths(audio_dir)
    print(f"Found {len(audio_paths)} audio files.")

    for path in tqdm(audio_paths, desc="Computing log-mel features"):
        try:
            audio, file_sr = soundfile.read(path)
            if file_sr != sr:
                continue  # Skip mismatched sampling rates
            mel = mel_spectrogram(
            y=torch.FloatTensor(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=80,
            sampling_rate=22050,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=8000
        )
            mel = mel.squeeze(0).numpy()  # shape: [80, T]
            all_logmel_values.append(mel)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    # Flatten across time and frequency
    print(f"Total log-mel values collected: {len(all_logmel_values)}")
    all_values = np.concatenate([m.flatten() for m in all_logmel_values])

    mean = np.mean(all_values)
    std = np.std(all_values)

    print(f"\n✅ Finished computing statistics:")
    print(f"→ Mean: {mean:.4f}")
    print(f"→ Std: {std:.4f}")

    return mean, std

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_audio_dir", type=str, default="/home/airis_lab/MJ/RIRLDM/datasets_mel_subset_complete/train_B", help="Path to training audio directory (e.g., /path/to/train_B)")
    parser.add_argument("--train_audio_dir", type=str, default="datasets/train_B", help="Path to training audio directory (e.g., /path/to/train_B)")
    args = parser.parse_args()

    compute_mean_std(args.train_audio_dir)
