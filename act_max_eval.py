import os

import torch
import torch.nn as nn
import torchvision.models as models

# Reading images
from torchvision import transforms
from PIL import Image
from numpy import asarray, percentile, tile

# Gaussian Kernel
from scipy.ndimage import gaussian_filter

import numpy as np
import matplotlib.pyplot as plt
import torchaudio.transforms

def plot_mel_spectrogram(mel_spectrogram, title='Mel Spectrogram', save_path='mel_spectrogram.png', dB_scale=True):
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

    plt.figure(figsize=(4, 8))
    plt.imshow(mel_spectrogram, origin="lower", aspect="auto", cmap="magma")
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency Bins")
    if dB_scale:
        plt.colorbar(format="%+2.0f dB")
    else:
        plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory
    
def plot_activation(activation, step, save_path='activation.png'):
    plt.bar(np.arange(len(activation.detach())), activation.detach())
    plt.title(f'Average activation pattern at step {step}')
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

def plot_waveform(waveform, sample_rate, title='Waveform', save_path='waveform.png'):
    """
    Plots a waveform using matplotlib.

    Args:
        waveform (Tensor): Tensor of shape [channels, time].
        sample_rate (int): Sample rate of the audio.
        title (str): Plot title.
        save_path (str): Path to save the figure.
    """
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    plt.figure(figsize=(10, 4))
    for i in range(num_channels):
        plt.plot(time_axis, waveform[i], label=f'Channel {i + 1}')
    
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    if num_channels > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

"""
Create a hook into target layer
    Example to hook into classifier 6 of Alexnet:
        alexnet.classifier[6].register_forward_hook(layer_hook('classifier_6'))
"""
def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[layer_name] = output
    return hook

"""
Reguarlizer, crop by absolute value of pixel contribution
"""
def abs_contrib_crop(img, threshold=0):

    abs_img = torch.abs(img)
    smalls = abs_img < percentile(abs_img, threshold)
    
    return img - img*smalls

"""
Regularizer, crop if norm of pixel values below threshold
"""
def norm_crop(img, threshold=0):

    norm = torch.norm(img, dim=0)
    norm = norm.numpy()

    # Create a binary matrix, with 1's wherever the pixel falls below threshold
    smalls = norm < percentile(norm, threshold)
    smalls = tile(smalls, (3,1,1))

    # Crop pixels from image
    crop = img - img*smalls
    return crop

"""
Optimizing Loop
    Dev: maximize layer vs neuron
"""
def act_max(network, 
    input, 
    layer_activation, 
    layer_name, 
    units, 
    steps=5, 
    alpha=torch.tensor(100), 
    update_viz=False,
    L2_Decay=False, 
    theta_decay=0.1,
    Gaussian_Blur=False,
    theta_every=20,
    theta_width=1,
    verbose=False,
    Norm_Crop=False,
    theta_n_crop=30,
    Contrib_Crop=False,
    theta_c_crop=30,
    z_k=None,
    Bin_Avg=False,
    beta=1.0,
    eps_cos=0.05,
    eps_mag=0.05,
    save_path='./visualisations'
    ):

    best_activation = -float('inf')
    best_img = input
    
    for step in range(steps):

        input.retain_grad() # non-leaf tensor
        # network.zero_grad()
        
        # Propogate image through network,
        # then access activation of target layer
        _ = network.encoder(input) # NOTE: don't need consuming encode() method
        layer_out = layer_activation[layer_name]
    
        # compute gradients w.r.t. target units,
        # then access the gradient of input (image) w.r.t. target units (neuron)
        
        # Average over time dimension
        latent_means = layer_out[0][units]  # (128, T_latent)
        avg = latent_means.mean(dim=-1)       # shape [128]
        
        # Norms of act and z_k
        avg_norm = avg.norm()
        k = z_k.norm()

        # Squared deviation from z_k magnitude 
        mag_div = ( avg_norm -  k ) ** 2

        # Cosine similarity to z_k
        cos_sim = torch.dot(avg, z_k) / (avg_norm * k + 1e-8)
        
        # Check stopping criterion
        if cos_sim > 1 - eps_cos and avg_norm - k < eps_mag:
            if verbose:
                print(f'Stopping criterion met at step {step}: cos_sim={cos_sim:.4f}, mag_div={mag_div:.4f}')
            break
        
        # Compute objective
        act = cos_sim - beta * mag_div

        # Backpropagate to input spectrogram
        act.backward(retain_graph=True)
        img_grad = input.grad
        
        # Gradient Step
        input = torch.add(input, torch.mul(img_grad, alpha))
        
        # Image space regularization does not contribute towards gradient
        """
        DEV:
            Detach input here
        """
        with torch.no_grad():

            # Regularization: L2
            if L2_Decay:
                input = torch.mul(input, (1.0 - theta_decay))

            # Regularization: Gaussian Blur
            if Gaussian_Blur and step % theta_every is 0:
                temp = input.squeeze(0)
                temp = temp.detach().numpy()
                cimg = gaussian_filter(temp, theta_width)
                input = torch.from_numpy(cimg).unsqueeze(0)

            # Regularization: Clip Norm
            if Norm_Crop:
                input = norm_crop(input.detach().squeeze(0), threshold=theta_n_crop)
                input = input.unsqueeze(0)

            # Regularization: Clip Contribution
            if Contrib_Crop:
                input = abs_contrib_crop(input.detach().squeeze(0), threshold=theta_c_crop)
                input = input.unsqueeze(0)
                
            if Bin_Avg and step % theta_every is 0: # experimental
                mel_avg = input.mean(dim=2, keepdim=True)
                input = mel_avg.expand(-1, -1, input.size(2))

        input.requires_grad_(True)

        if verbose:
            print('step: ', step, 'activation: ', act, 'cos_sim: ', cos_sim, 'norm: ', avg_norm)
        
        if update_viz:
            spec = input.detach()
            plot_mel_spectrogram(spec, title=f'Optimal Mel Spectrogram at step {step}', save_path=os.path.join(save_path, f'final_mel.png'), dB_scale=True)
            plot_activation(avg, step, save_path=os.path.join(save_path, f'final_activation.png'))
            # plot_waveform(input.detach().squeeze(0), sample_rate=44100, title='Optimal Waveform', save_path='./visualisations/final_waveform.png')
            # if step % 50 == 0:
            #     torch.save(latent_means, f'./visualisations/activation_pattern_step_{step}.pt')

        # Keep highest activation
        if best_activation < act:
            best_activation = act
            best_img = input

    return best_img