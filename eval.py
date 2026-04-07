import os
import sys
# sys.path.append(os.path.abspath('.'))
import act_max_eval as amu
import torch
import torchaudio
import gin
import rave
import matplotlib.pyplot as plt
import numpy as np
import types
import torch.nn.functional as F
import librosa

# Model directory path
model_dir = './checkpoints/complex/drumset'
config_path = rave.core.search_for_config(model_dir)
gin.parse_config_file(config_path)
model = rave.RAVE()
run = rave.core.search_for_run(model_dir)
model = model.load_from_checkpoint(run)

# Parameters for activation maximization
steps = 5               # perform 100 iterations
units = range(model.latent_size)     # take all latents
alpha = torch.tensor(0.00001)   # learning rate (step size)
verbose = True              # print activation every step
L2_Decay = False             # enable L2 decay regularizer
Gaussian_Blur = True        # enable Gaussian regularizer
Norm_Crop = False            # enable norm regularizer
Contrib_Crop = False         # enable contribution regularizer
Bin_Avg = False              # [Experimental] enable (some kind of) bin averaging of mel spectrogram

def generate_input_mel(model, z_k):
    
    num_frames = 8 # compression ratio of encoder
    initial_mel = torch.ones(1, 128, num_frames) * 0.005 #TODO: use kaiming yeah?
    
    output_layer_encoder = model.encoder.encoder.net[-1]
    layer_name = 'output_layer_encoder'
    act_dict = {}
    output_layer_encoder.register_forward_hook(amu.layer_hook(act_dict, layer_name))

    input = torch.log1p(initial_mel) # log scaling like in RAVE _mel_encode()
    input.requires_grad_(True)  # enable gradient computation
    
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
        z_k=z_k
    )
    return output
    

def generate_output_mel(model, z_k):
    z_k = z_k.unsqueeze(0).unsqueeze(-1) # add batch and time dimensions: [latent] -> [1, latent, 1]
    output_chunk = model.decode(z_k)
    return model.spectrogram(output_chunk.squeeze(0).detach())

def sc_from_mel(mel_spec, freqs):
    # mel_spec: [1, n_mels, time] or [n_mels, time]
    freqs = freqs.view(1, -1, 1) if mel_spec.dim() == 3 else freqs.view(-1, 1)
    return (freqs * mel_spec).sum(dim=-2) / (mel_spec.sum(dim=-2) + 1e-8)


def run_evaluation():
    
    print(f'Model loaded from {model_dir}')
    print(f'model.latent_size: {model.latent_size}')
    
    # Define test points along pca axis
    k_vals = [-2, -1, 0, 1, 2]
    
    num_pca = 3
    for i in range(num_pca):
        pc = model.latent_pca[i]
        assert torch.norm(pc) == 1, f'PCA vector {i} is not normalized'
        for k in k_vals:
            print(f'\nEvaluating PCA component {i} at k={k}...')
            z_k = pc * k
            input_mel = generate_input_mel(model, z_k) # Run ActMax
            output_mel = generate_output_mel(model, z_k) # Run decoder

            # TODO: compute freq-domain AFs
            
            # Spectral centroid from mel spectrogram
            mel_freqs = torch.tensor(
                librosa.mel_frequencies(n_mels=input_mel.shape[-2], fmin=0, fmax=model.sr / 2),
                dtype=torch.float32
            )  # shape: [n_mels]
            sc_input = sc_from_mel(input_mel, mel_freqs)
            sc_output = sc_from_mel(output_mel, mel_freqs)
            
            
            breakpoint()
    

if __name__ == "__main__":
    run_evaluation()