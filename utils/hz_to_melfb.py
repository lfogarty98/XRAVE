import act_max_util as amu
import torch
import torchaudio
import gin
import rave
import matplotlib.pyplot as plt
import numpy as np
import types


# Directory containing model checkpoint and config
model_dir = '/Users/DiarmuidFogarty/repos/XRAVE/checkpoints/fivesines'

# Load the model
config_path = rave.core.search_for_config(model_dir)
gin.parse_config_file(config_path)
model = rave.RAVE()
run = rave.core.search_for_run(model_dir)
model = model.load_from_checkpoint(run)

### Code to find the mel bin closest to a target frequency ###

# Model's mel spectrogram parameters
sr = model.spectrogram.sample_rate
n_fft = model.spectrogram.n_fft
n_mels = model.spectrogram.n_mels
f_min = 0.0
f_max = sr / 2

# Create mel filterbank matrix
mel_fb = torchaudio.functional.melscale_fbanks(
    n_freqs=(n_fft // 2) + 1,
    f_min=f_min,
    f_max=f_max,
    n_mels=n_mels,
    sample_rate=sr,
    norm=None,  # or "slaney" if used
    mel_scale="htk"  # or "slaney" depending on your config
)

# Frequencies for each FFT bin
fft_freqs = torch.linspace(0, sr / 2, (n_fft // 2) + 1)
fft_freqs = fft_freqs[:, None]


# Center freq for each mel bin = weighted average of FFT bin freqs using mel filter weights
mel_freqs = (mel_fb * fft_freqs).sum(dim=0) / mel_fb.sum(dim=0)

# Find bin closest to target frequency
target_freq = 290
closest_bin = torch.argmin(torch.abs(mel_freqs - target_freq))

print(f"Closest mel bin to {target_freq} Hz is bin {closest_bin.item()}, "
      f"center freq ≈ {mel_freqs[closest_bin]:.2f} Hz")