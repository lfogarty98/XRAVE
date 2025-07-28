import sys, os 
sys.path.append(os.path.abspath('.'))

import torch
import gin
import rave
import types
import act_max_util as amu


# Directory containing model checkpoint and config
model_dir = './experiments/fivesines_15min'

# Load the model
config_path = rave.core.search_for_config(model_dir)
gin.parse_config_file(config_path)
model = rave.RAVE()
run = rave.core.search_for_run(model_dir)
model = model.load_from_checkpoint(run)

# sanity check: mel bins matches encoder input layer channels
input_layer_encoder = model.encoder.encoder.net[0]
in_channels = input_layer_encoder.in_channels
n_mels = model.spectrogram.mel_scale.n_mels
assert in_channels == n_mels, \
    f'Input layer channels {in_channels} do not match melspectrogram channels {n_mels}.'

# create random melspectrogram from random waveform
T = 4000 # NOTE: T affects the number of stacked latent vectors [128, T_latent]. 2000 leads to T_latent = 1
random_audio = torch.randn(1, T)  # shape: (C, T)
initial_mel = model.spectrogram(random_audio) # shape: (n_mels, T_mel) 

# Plot the initial mel spectrogram
amu.plot_mel_spectrogram(initial_mel, title='Initial Mel Spectrogram', save_path='visualisations/initial_mel.png')

# Prepare input for activation maximization
input = initial_mel 
input.requires_grad_(True)  # enable gradient computation

# Create hook into target layer
# NOTE: Target layer is the input layer of the decoder, at which point the latent vector has been 
# been computed via the sampling step in VariationalEncoder.reparametrize()
input_layer_decoder = model.decoder.net[0] 
# output_layer_encoder = model.encoder.encoder.net[-1]  # TODO: (ideally) take means from actual encoder output

act_dict = {}
layer_name = 'input_layer_decoder'
# layer_name = 'output_layer_encoder'
input_layer_decoder.register_forward_hook(amu.layer_hook(act_dict, layer_name))

# We need to override the encode method, since we assume direct melspec input
def encode_melspec_only(self, x, return_mb: bool = False):
    z = self.encoder(x)
    return z
model.encode = types.MethodType(encode_melspec_only, model)

# Parameters for activation maximization
steps = 100                 # perform 100 iterations
unit = 0                  # first latent dimension (arbitrary ordering?) # TODO: use PCA-ordering of latents
alpha = torch.tensor(100)   # learning rate (step size) 
verbose = True              # print activation every step
L2_Decay = False             # enable L2 decay regularizer
Gaussian_Blur = False        # enable Gaussian regularizer
Norm_Crop = False            # enable norm regularizer
Contrib_Crop = False         # enable contribution regularizer

breakpoint()

# Run activation maximization
output = amu.act_max(network=model,
                input=input,
                layer_activation=act_dict,
                layer_name=layer_name,
                unit=unit,
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