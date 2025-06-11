import sys, os 
sys.path.append(os.path.abspath('.'))

import torch
import gin
import rave
import act_max_util as amu


# Directory containing model checkpoint and config
model_dir = '/Users/DiarmuidFogarty/repos/RAVE/jelinek'

# Load the model
config_path = rave.core.search_for_config(model_dir)
gin.parse_config_file(config_path)
model = rave.RAVE()
run = rave.core.search_for_run(model_dir)
model = model.load_from_checkpoint(run)

# extract output layer of encoder
wasserstein_encoder = model.encoder
output_layer_encoder = wasserstein_encoder.encoder.net[-1]

breakpoint()

# create a dummy input
input = torch.randn(1, 1, 16000)  # Example shape for a single audio sample
input.requires_grad_(True)

# NOTE: input goes through PQMF block before being passed to the encoder (check model.pqmf)
# pqmf = model.pqmf
# x_encoded_input = pqmf.forward_conv(x)
# print(x_encoded_input.shape)


# Create hook into target layer
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