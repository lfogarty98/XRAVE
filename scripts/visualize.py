import sys, os 
sys.path.append(os.path.abspath('.'))

import torch
import numpy as np
import gin
import rave
import types
import act_max_util as amu


# Directory containing model checkpoint and config
# model_dir = './checkpoints/fivesines_15min'
# model_dir = './checkpoints/jelinek_mel_b05386f725/version_0'
# model_dir = './checkpoints/bach'
# model_dir = './checkpoints/sweptsines'
# model_dir = './checkpoints/twosines'
model_dir = './checkpoints/fivesines'

# Load the model
config_path = rave.core.search_for_config(model_dir)
gin.parse_config_file(config_path)
model = rave.RAVE()
run = rave.core.search_for_run(model_dir)
model = model.load_from_checkpoint(run)




# create random melspectrogram from random waveform
T = 2000 # NOTE: T affects the number of stacked latent vectors [128, T_latent]. 2000 leads to T_latent = 1
random_audio = torch.randn(1, T)  # shape: (C, T)
initial_mel = model.spectrogram(random_audio) # shape: (n_mels, T_mel) 

# Plot the initial mel spectrogram
amu.plot_mel_spectrogram(initial_mel, title='Initial Mel Spectrogram', save_path='visualisations/initial_mel.png')

# Prepare input for activation maximization
input = initial_mel
input.requires_grad_(True)  # enable gradient computation

# Create hook into target layer
output_layer_encoder = model.encoder.encoder.net[-1]  # NOTE: (means, scales): (128,128)
layer_name = 'output_layer_encoder'
act_dict = {}
output_layer_encoder.register_forward_hook(amu.layer_hook(act_dict, layer_name))

# We need to override the encode method, since we assume direct melspec input
def encode_melspec_only(self, x, return_mb: bool = False):
    z = self.encoder(x)
    return z
model.encode = types.MethodType(encode_melspec_only, model)

# Extract the first principal component of the latent space and plot it
top_pc1 = model.latent_pca[0] # NOTE: sign matters!
import matplotlib.pyplot as plt
plt.bar(np.arange(len(top_pc1)), top_pc1)
plt.title('Top Principal Component (PC1)')
plt.savefig('visualisations/latent_pca_top_pc1.png')
plt.close()

# Select the top N dimensions to maximize
N = 15
important_dims = torch.argsort(top_pc1, descending=True)
important_dims = important_dims[:N]  # Select top N dimensions


# Parameters for activation maximization
steps = 5000                # perform 100 iterations
units = important_dims     # take latents contributing the most to the first PC
alpha = torch.tensor(100)   # learning rate (step size) 
verbose = True              # print activation every step
L2_Decay = False             # enable L2 decay regularizer
Gaussian_Blur = False        # enable Gaussian regularizer
Norm_Crop = False            # enable norm regularizer
Contrib_Crop = False         # enable contribution regularizer


# Run activation maximization
output = amu.act_max(network=model,
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
                )

amu.plot_mel_spectrogram(output.detach(), title='Optimal Mel Spectrogram', save_path='visualisations/final_mel.png')


breakpoint()