import sys, os 
sys.path.append(os.path.abspath('.'))

import torch
import gin
import rave
import act_max_util as amu


# Directory containing model checkpoint and config
model_dir = '/media/dc-04-vol03/liam/XRAVE/experiments/jelinek_mel_b05386f725'

# Load the model
config_path = rave.core.search_for_config(model_dir)
gin.parse_config_file(config_path)
model = rave.RAVE()
run = rave.core.search_for_run(model_dir)
model = model.load_from_checkpoint(run)

# extract melspectrogram from model
melspectrogram = model.spectrogram
n_mels = melspectrogram.n_mels

# extract input layer of encoder
input_layer_encoder = model.encoder.encoder.net[0]
in_channels = input_layer_encoder.in_channels
assert in_channels == melspectrogram.n_mels, \
    f'Input layer channels {in_channels} do not match melspectrogram channels {n_mels}.'

# create random melspectrogram
# initial_mel = torch.randn(1, n_mels, 16000)  # shape: (B, C, T)

random_audio = torch.randn(1, 16000)  # shape: (C, T)
initial_mel = model.spectrogram(random_audio)

# Plot the initial mel spectrogram
amu.plot_mel_spectrogram(initial_mel, title='Initial Mel Spectrogram')


breakpoint()

# Prepare input for activation maximization
input = initial_mel.unsqueeze(0)  # add batch dimension
input.requires_grad_(True)  # enable gradient computation

breakpoint()

# Create hook into target layer
output_layer_encoder = model.encoder.encoder.net[-1]
act_dict = {}
layer_name = 'output_layer_encoder'
output_layer_encoder.register_forward_hook(amu.layer_hook(act_dict, layer_name))

# Parameters for activation maximization
steps = 100                 # perform 100 iterations
unit = 130                  # flamingo class of Imagenet
alpha = torch.tensor(100)   # learning rate (step size) 
verbose = True              # print activation every step
L2_Decay = True             # enable L2 decay regularizer
Gaussian_Blur = True        # enable Gaussian regularizer
Norm_Crop = True            # enable norm regularizer
Contrib_Crop = True         # enable contribution regularizer

# Run activation maximization
output = amu.act_max(network=model.encoder.encoder, # only encoder
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

breakpoint()