import sys, os 
sys.path.append(os.path.abspath('.'))

import torch
import gin
import rave
import types
import act_max_util as amu


# Directory containing model checkpoint and config
model_dir = '/Users/DiarmuidFogarty/repos/XRAVE/experiments/jelinek_mel_b05386f725'

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



# Prepare input for activation maximization
input = initial_mel 
input.requires_grad_(True)  # enable gradient computation

# Create hook into target layer
# Target layer is the input layer of the decoder, at which point the latent vector has been 
# been computed via the sampling step in VariationalEncoder.reparametrize()
input_layer_decoder = model.decoder.net[0]
act_dict = {}
layer_name = 'input_layer_decoder'
input_layer_decoder.register_forward_hook(amu.layer_hook(act_dict, layer_name))

# We need to override the encode method, since we assume direct melspec input
def encode_melspec_only(self, x, return_mb: bool = False):
    z = self.encoder(x)
    return z

model.encode = types.MethodType(encode_melspec_only, model)

# Parameters for activation maximization
steps = 100                 # perform 100 iterations
unit = 0                  # flamingo class of Imagenet
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