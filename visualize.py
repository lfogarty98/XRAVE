#!/usr/bin/env python3
"""
Visualize Latents with ActMax
Converted from visualize.ipynb - saves figures instead of displaying them
"""

import sys
import os
from pathlib import Path

import torch
import torchaudio
import numpy as np
import gin
import rave
import types
# import act_max_util as amu
import act_max_pca as amu
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Create output directory for figures
OUTPUT_DIR = Path('./figures')
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Figures will be saved to: {OUTPUT_DIR.absolute()}")

# ============================================================================
# Plotting functions
# ============================================================================

def plot_mel_spectrogram(mel_spectrogram, title='Mel Spectrogram', dB_scale=True, save_path=None):
    # Remove batch dimension if present
    if mel_spectrogram.dim() == 3 and mel_spectrogram.size(0) == 1:
        mel_spectrogram = mel_spectrogram[0]

    if dB_scale:
        mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

    else:
        # Normalize mel spectrogram to [0, 1] range
        mel_min = mel_spectrogram.min()
        mel_max = mel_spectrogram.max()
        if mel_max == mel_min:  # Avoid division by zero
            mel_spectrogram = mel_spectrogram - mel_min
        else:     # Normalize to [0, 1]
            mel_spectrogram = (mel_spectrogram - mel_min) / (mel_max - mel_min)

    plt.figure(figsize=(10, 7))
    plt.imshow(mel_spectrogram, origin="lower", aspect="auto", cmap="magma")
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency Bins")
    if dB_scale:
        plt.colorbar(format="%+2.0f dB")
    else:
        plt.colorbar(label="Amplitude")
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_latent_means(latent_means, title='Latent Means', save_path=None):
    """
    Plots the average value of each latent mean across time frames.

    Args:
        latent_means (Tensor): Tensor of shape [1, channels, time].
        title (str): Plot title.
        save_path (str): Path to save the figure.
    """
    averages = latent_means.squeeze(0).mean(dim=1).detach()  # Average over time frames
    plt.bar(np.arange(len(averages)), averages)
    plt.title(title)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Average Value')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()


def plot_latent_trajectories(
    latent_means,
    title='Latent Means Trajectories',
    save_path=None
):
    """
    Plots the trajectories of latent means over time using step plots.

    Args:
        latent_means (Tensor): Tensor of shape [1, channels, time].
        title (str): Plot title.
        save_path (str): Path to save the figure.
    """
    latent_np = latent_means.squeeze(0).cpu().detach().numpy()
    T = latent_np.shape[1]  # number of timesteps

    plt.figure(figsize=(10, 6))
    for dim in range(latent_np.shape[0]):
        plt.step(
            range(T),
            latent_np[dim, :],
            where='mid',
            alpha=0.8,
            label=f'Dim {dim}'
        )

    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('Latent Mean Value')
    # plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================================
# Load model
# ============================================================================

# Directory containing model checkpoint and config
model_dir = './checkpoints/complex/drumset'

# Load the model
config_path = rave.core.search_for_config(model_dir)
gin.parse_config_file(config_path)
model = rave.RAVE()
run = rave.core.search_for_run(model_dir)
model = model.load_from_checkpoint(run)


print(f'Model loaded from {model_dir}')
print(f'model.latent_size: {model.latent_size}')
print(model)

# ============================================================================
# Initialize input
# ============================================================================

num_frames = 8 # compression ratio of encoder
initial_mel = torch.ones(1, 128, num_frames) * 0.005

# from torch.nn.init import kaiming_normal_
# torch.manual_seed(42)  # for reproducibility
# initial_mel = torch.empty(1, 128, num_frames)
# kaiming_normal_(initial_mel, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
# initial_mel *= 0.05
# breakpoint()

plot_mel_spectrogram(initial_mel, title='Initial Mel Spectrogram', save_path=OUTPUT_DIR / '01_initial_mel.png')

# ============================================================================
# Plot PCA component
# ============================================================================

N = 0
pc = model.latent_pca[N]

plt.bar(np.arange(len(pc)), pc)
plt.title(f'{N+1}th Principal Component')
plt.xlabel('Latent Dimension')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_pca_component.png')
print(f"Saved: {OUTPUT_DIR / '02_pca_component.png'}")
plt.close()

# ============================================================================
# Create hook into final encoder layer
# ============================================================================

# Create hook into target layer
output_layer_encoder = model.encoder.encoder.net[-1]  # NOTE: (means, scales): (128,128)
layer_name = 'output_layer_encoder'
act_dict = {}
output_layer_encoder.register_forward_hook(amu.layer_hook(act_dict, layer_name))

# ============================================================================
# Prepare optimization
# ============================================================================

input = torch.log1p(initial_mel) # log scaling like in RAVE _mel_encode()
input.requires_grad_(True)  # enable gradient computation

# Parameters for activation maximization
steps = 200               # perform 100 iterations
units = range(model.latent_size)     # take all latents
alpha = torch.tensor(0.00001)   # learning rate (step size)
verbose = True              # print activation every step
L2_Decay = False             # enable L2 decay regularizer
Gaussian_Blur = True        # enable Gaussian regularizer
Norm_Crop = False            # enable norm regularizer
Contrib_Crop = False         # enable contribution regularizer
Bin_Avg = False              # [Experimental] enable (some kind of) bin averaging of mel spectrogram

pc = -pc # choose direction of PC to maximize

# ============================================================================
# Run ActMax
# ============================================================================

# Run activation maximization
output = amu.act_max(
    network=model,
    input=input,
    layer_activation=act_dict,
    layer_name=layer_name,
    units=units,
    steps=steps,
    alpha=alpha,
    verbose=verbose,
    L2_Decay=L2_Decay,
    Gaussian_Blur=Gaussian_Blur,
    Norm_Crop=Norm_Crop,
    Contrib_Crop=Contrib_Crop,
    update_viz=True,
    theta_width=1,
    theta_every=4,
    pc=pc
)

# ============================================================================
# Check latent trajectory for concatenated output
# ============================================================================
breakpoint()

# [Experimental] Try setting each entry to the average of the amplitudes across all time frames
# mel_avg = output.mean(dim=2, keepdim=True)
# mel_flat = mel_avg.expand(-1, -1, output.size(2))
# plot_mel_spectrogram(mel_flat.detach(), title='Flattened Mel', dB_scale=False, save_path=OUTPUT_DIR / '03_flattened_mel.png')
# print(output.shape)

# print(output[:, 0, :])
# print(output[:, 127, :])

# print(torch.mean(output[:, 0, :]))
# print(torch.mean(output[:, 127, :]))

# print(torch.mean(mel_flat[:, 0, :]))
# print(torch.mean(mel_flat[:, 127, :]))
# output = mel_flat

# Concatenate output multiple times to see latent trajectory
n = 10
output_concat = torch.cat([output]*n, dim=-1)
# output_concat = F.pad(output, (10, 10), mode="constant", value=1)
z_enc = model.encoder(output_concat)
means, logscales = torch.split(z_enc, z_enc.shape[1] // 2, 1)
plot_mel_spectrogram(output_concat.detach(), title='Concatenated Mel Output', dB_scale=False, save_path=OUTPUT_DIR / '04_concatenated_mel.png')

plot_latent_means(means, title='Latent Means after Activation Maximization', save_path=OUTPUT_DIR / '05_latent_means.png')

plot_latent_trajectories(means, title='Latent Means Trajectories after Activation Maximization', save_path=OUTPUT_DIR / '06_latent_trajectories.png')

print(f"\nAll figures saved to: {OUTPUT_DIR.absolute()}")
