import os
import sys
sys.path.append(os.path.abspath('.'))

import act_max_util as amu
import torch
import torchaudio
import gin
import rave
import matplotlib.pyplot as plt
import numpy as np
import types

    
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
    
def plot_latent_trajectories(latent_means, title='Latent Means Trajectories', save_path='latent_trajectories.png'):
    """
    Plots the trajectories of latent means over time.

    Args:
        latent_means (Tensor): Tensor of shape [1, channels, time].
        title (str): Plot title.
        save_path (str): Path to save the figure.
    """
    latent_np = latent_means.squeeze(0).cpu().detach().numpy()
    T = latent_np.shape[1]  # number of timesteps
    for dim in range(latent_np.shape[0]):
        plt.plot(range(T), latent_np[dim, :], alpha=0.6)
    
    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('Latent Means')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

def plot_latent_means(latent_means, title='Latent Means', save_path='latent_means.png'):
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
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

def plot_principal_component(pc, title='Principal Component', save_path='principal_component.png'):
    """
    Plots a principal component.

    Args:
        pc (Tensor): Principal component tensor.
        title (str): Plot title.
        save_path (str): Path to save the figure.
    """
    plt.bar(np.arange(len(pc)), pc)
    plt.title(title)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

    
# Directory containing model checkpoint and config
model_dir = '/Users/DiarmuidFogarty/repos/XRAVE/checkpoints/foursines_warmup_200k'

# Load the model
config_path = rave.core.search_for_config(model_dir)
gin.parse_config_file(config_path)
model = rave.RAVE()
run = rave.core.search_for_run(model_dir)
model = model.load_from_checkpoint(run)

# We need to override the encode method, since we assume direct melspec input
def encode_melspec_only(self, x, return_mb: bool = False):
    z = self.encoder(x)
    return z
model.encode = types.MethodType(encode_melspec_only, model)    



# Load a sine wave
sine_wave, sr = torchaudio.load('./training_data/fivesines_15min/sines_signal_3.wav')
# sine_wave, sr = torchaudio.load('/Users/DiarmuidFogarty/repos/test-signals-dataset/sweptsines_15min/swept_sines_signal_54.wav') # NOTE: maybe interesting for looking at latent trajectories

# Take a chunk
T_chunk = 128000  # Number of samples to take NOTE: T affects the number of stacked latent vectors [128, T_latent]. 2000 leads to T_latent = 1
sine_chunk = sine_wave[:, :T_chunk]  # Take the first T_chunk samples
plot_waveform(sine_chunk, sr, title='Input Waveform Chunk', save_path='utils/viz/input.png')

# Create mel spectrogram from sine wave chunk
melspec_sine = model.spectrogram(sine_chunk)
amu.plot_mel_spectrogram(melspec_sine, title='Mel Spectrogram of Sine Wave', save_path='utils/viz/input_mel.png')

# Pass melspec chunk to encoder and save latent means
z = model.encode(melspec_sine, return_mb=False)
latent_means = z[:, :128, :] # (c, 256, T_latent)


# Plot the trajectories of the latent means
plot_latent_trajectories(latent_means, title='Latent Means Trajectories', save_path='utils/viz/latent_trajectories.png')

# Plot the average value of each latent mean across time frames
plot_latent_means(latent_means, title='Average Latent Means', save_path='utils/viz/latent_means.png')



### Sanity check: manually set latent means w.r.t principal components and decode back to mel spectrogram ###

# Choose Nth principal component and plot it take the top (max or min) M dimensions
N = 0
pc = model.latent_pca[N]
plot_principal_component(pc, title=f'{N+1}th Principal Component', save_path=f'utils/viz/foursines_200k/pc{N+1}.png')

# Modify latent means to weigh by the principal component
latent_means[:] = 1 
pca_latent_means = latent_means * pc[None, :, None] # element-wise multiplication keeping batch dimension and time dimension

# Walk the latent means along the PC axis
alpha = 0.5  # scaling factor corresponding to parameter offset
pca_latent_means_scaled = pca_latent_means * alpha

# Feed modified latent means back to decoder and plot the reconstructed waveform
output_chunk = model.decode(pca_latent_means_scaled).detach().cpu()
output_chunk = output_chunk.squeeze(0)  # Remove batch dimension
plot_waveform(output_chunk, sr, title='Reconstructed Waveform from Latent Means', save_path='utils/viz/output.png')


# Plot the (mel or stft) spectrogram of the reconstructed waveform
# reconstructed_stft = model.spectrogram.spectrogram(output_chunk)
# amu.plot_mel_spectrogram(
#     reconstructed_stft, 
#     title=f'Reconstructed Spectrogram from Latent Means (N={N})', 
#     save_path='utils/viz/output_spec3.png', 
#     dB_scale=True
# )

reconstructed_mel = model.spectrogram(output_chunk)
amu.plot_mel_spectrogram(
    reconstructed_mel, 
    title=f'Reconstructed Mel Spectrogram from Latent Means', 
    save_path=f'utils/viz/foursines_200k/output_mel_pca1_alpha_{alpha}.png', 
    dB_scale=True
)