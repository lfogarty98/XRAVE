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
import pandas as pd

# Model directory path
model_dir = './checkpoints/complex/drumset'
config_path = rave.core.search_for_config(model_dir)
gin.parse_config_file(config_path)
model = rave.RAVE()
run = rave.core.search_for_run(model_dir)
model = model.load_from_checkpoint(run)

# Parameters for activation maximization
steps = 500               # perform 500 iterations
units = range(model.latent_size)     # take all latents
alpha = torch.tensor(0.00001)   # learning rate (step size)
verbose = True              # print activation every step
L2_Decay = False             # enable L2 decay regularizer
Gaussian_Blur = True        # enable Gaussian regularizer
Norm_Crop = False            # enable norm regularizer
Contrib_Crop = False         # enable contribution regularizer
Bin_Avg = False              # [Experimental] enable (some kind of) bin averaging of mel spectrogram

def generate_input_mel(model, z_k, save_path='./visualisations'):
    
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
        z_k=z_k,
        eps_cos=0.05,
        eps_mag=0.05,
        save_path=save_path
    )
    return output
    

def generate_output_mel(model, z_k):
    z_k = z_k.unsqueeze(0).unsqueeze(-1) # add batch and time dimensions: [latent] -> [1, latent, 1]
    output_chunk = model.decode(z_k)
    return model.spectrogram(output_chunk.squeeze(0).detach())

def spectral_features_from_mel(mel_spec, freqs):
    # Computes mean spectral centroid and bandwidth averaged over time.
    # mel_spec: [1, n_mels, time] or [n_mels, time]
    # freqs:    [n_mels]
    # returns:  (centroid, bandwidth) as scalar tensors
    freqs_exp = freqs.view(1, -1, 1) if mel_spec.dim() == 3 else freqs.view(-1, 1)
    power_sum = mel_spec.sum(dim=-2) + 1e-8                          # [..., time]
    centroid = (freqs_exp * mel_spec).sum(dim=-2) / power_sum        # [..., time]
    c_exp = centroid.unsqueeze(1) if mel_spec.dim() == 3 else centroid.unsqueeze(0)
    bandwidth = torch.sqrt(((freqs_exp - c_exp) ** 2 * mel_spec).sum(dim=-2) / power_sum)
    return centroid.mean(), bandwidth.mean()


def run_evaluation():
    
    print(f'Model loaded from {model_dir}')
    print(f'model.latent_size: {model.latent_size}')
    
    # Define test points along pca axis
    k_vals = [-2, -1, 0, 1, 2]
    
    # Number of PCA components to evaluate
    num_pca = 1
    
    # Create evaluation directory
    import time
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    eval_dir = f'./visualisations/evaluation_{timestamp}/'
    
    results = []

    for i in range(num_pca):
        pc = model.latent_pca[i]
        assert torch.norm(pc) == 1, f'PCA vector {i} is not normalized'
        for k in k_vals:
            z_k = pc * k

            dir_path = os.path.join(eval_dir, f'pca_component_{i}_k_{k}')# first idx is pca component, second idx is k value
            os.makedirs(dir_path, exist_ok=True)
            print(f'\nEvaluating PCA component {i} at k={k}...')

            # Generate input and output spectra
            input_mel = generate_input_mel(model, z_k, save_path=dir_path) # Run ActMax
            output_mel = generate_output_mel(model, z_k) # Run decoder

            # Compute spectral features
            mel_freqs = torch.tensor(
                librosa.mel_frequencies(n_mels=input_mel.shape[-2], fmin=0, fmax=model.sr / 2),
                dtype=torch.float32
            )  # shape: [n_mels]
            sc_input, sb_input = spectral_features_from_mel(input_mel, mel_freqs)
            sc_output, sb_output = spectral_features_from_mel(output_mel, mel_freqs)

            # Save metrics & plots (input mel plot, output mel plot, final activation vector, AFs)
            results.append({
                'pca_component': i,
                'k': k,
                'sc_input':  sc_input.item(),
                'sc_output': sc_output.item(),
                'sb_input':  sb_input.item(),
                'sb_output': sb_output.item(),
            })
            amu.plot_mel_spectrogram(output_mel, title=f'Output Mel Spectrogram (PCA {i}, k={k})', save_path=os.path.join(dir_path, 'output_mel.png'), dB_scale=True)

    csv_path = os.path.join(eval_dir, 'results.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f'\Metrics saved to {csv_path}')
    

if __name__ == "__main__":
    run_evaluation()