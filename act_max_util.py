import torch
import torch.nn as nn
import torchvision.models as models

# Showing images
import cv2

# Reading images
from torchvision import transforms
from PIL import Image
from numpy import asarray, percentile, tile

# Gaussian Kernel
from scipy.ndimage import gaussian_filter

import numpy as np
import matplotlib.pyplot as plt
import torchaudio.transforms

def plot_mel(melspec, save_path='mel_spectrogram.png', title='Mel Spectrogram'):
    plt.imshow(melspec, origin="lower", aspect="auto", cmap="magma")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency Bins")
    plt.title(title)
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

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

# https://medium.com/analytics-vidhya/deep-dream-visualizing-the-features-learnt-by-convolutional-networks-in-pytorch-b7296ae3b7f
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
denormalize = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229, 1/0.224, 1/0.225] )
def image_converter(im):
    
    # move the image to cpu
    im_copy = im.cpu()
    
    # for plt.imshow() the channel-dimension is the last
    # therefore use transpose to permute axes
    im_copy = denormalize(im_copy.clone().detach()).numpy()
    im_copy = im_copy.transpose(1,2,0)
    
    # clip negative values as plt.imshow() only accepts 
    # floating values in range [0,1] and integers in range [0,255]
    im_copy = im_copy.clip(0, 1) 
    
    return im_copy

# https://stackoverflow.com/questions/50420168/how-do-i-load-up-an-image-and-convert-it-to-a-proper-tensor-for-pytorch
def get_image(img_path):
    img = Image.open(img_path) # use pillow to open a file
    img = img.resize((256, 256)) # resize the file to 256x256
    img = img.convert('RGB') #convert image to RGB channel

    img = asarray(img).transpose(-1, 0, 1) # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
    img = img/255
    img = torch.from_numpy(img) # create the image tensor
    return img

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
    generate_gif=False,
    update_viz=False,
    path_to_gif='./',
    L2_Decay=False, 
    theta_decay=0.1,
    Gaussian_Blur=False,
    theta_every=4,
    theta_width=1,
    verbose=False,
    Norm_Crop=False,
    theta_n_crop=30,
    Contrib_Crop=False,
    theta_c_crop=30,
    pc=None
    ):

    best_activation = float('inf')
    best_img = input
    
    # pc = pc / pc.norm()                 # normalize once outside the loop
    
    for k in range(steps):

        input.retain_grad() # non-leaf tensor
        # network.zero_grad()
        
        # Propogate image through network,
        # then access activation of target layer
        # output = network(input)
        _ = network.encoder(input) # NOTE: don't need consuming encode() method
        layer_out = layer_activation[layer_name]
        
        # NOTE: layer_out[0] has dim (128 (256), T_latent)
        
        # compute gradients w.r.t. target units,
        # then access the gradient of input (image) w.r.t. target units (neuron)
        
        # latent_means = layer_out[units]  # (128, T_latent)
        latent_means = layer_out[0][units]  # (128, T_latent)
        
        # latent_means_scaled = latent_means * pc[None, :, None]
        # act = latent_means_scaled.sum() # or mean??

        # # Average over time dimension
        avg = latent_means.mean(dim=-1)       # shape [128]
        
        # # Alignment at every timestep
        # cos_sims = torch.nn.functional.cosine_similarity(
        #     latent_means.transpose(0,1),   # shape [T_latent, 128]
        #     pc.unsqueeze(0),               # shape [1, 128]
        #     dim=-1
        # )  # shape [T_latent]
        # act = cos_sims.mean()


        # # Cosine similarity to PC
        dot = torch.dot(avg, pc)
        pc_normalized = pc / pc.norm() 
        cos_sim = torch.dot(avg, pc_normalized) / (avg.norm() + 1e-8)
        gamma = 1.0 # weighting factor for dot product
        beta = 5.0  # weighting factor for cosine similarity
        # if cos_sim < 0.8:
        #     act = cos_sim
        # else:
        #     # alpha = 1
        #     act = dot
        act = dot
        # act = gamma * dot + beta * cos_sim
        
        # act = layer_out[0][units].sum()  # sum activations of all target units
        unit = 7
        # act = gamma * layer_out[0][unit].mean(dim=-1) + beta * cos_sim
        act = layer_out[0][unit].mean(dim=-1)
        
        act.backward(retain_graph=True)
        img_grad = input.grad
        
        # Gradient Step
        # input = input + alpha * dimage_dneuron
        input = torch.subtract(input, torch.mul(img_grad, alpha))
        
        # regularization does not contribute towards gradient
        """
        DEV:
            Detach input here
        """
        with torch.no_grad():

            # Regularization: L2
            if L2_Decay:
                input = torch.mul(input, (1.0 - theta_decay))

            # Regularization: Gaussian Blur
            if Gaussian_Blur and k % theta_every is 0:
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

        input.requires_grad_(True)

        if verbose:
            print('step: ', k, 'activation: ', act, 'cos_sim: ', cos_sim)

        if generate_gif:
            frame = input.detach().squeeze(0)
            frame = image_converter(frame)
            frame = frame * 255
            cv2.imwrite(path_to_gif+str(k)+'.jpg', frame)
        
        if update_viz:
            # spec = network.spectrogram(input.detach().squeeze(0))
            # plot_mel_spectrogram(spec, title='Optimal Mel Spectrogram', save_path='./visualisations/final_mel.png')
            spec = input.detach()
            plot_mel_spectrogram(spec, title=f'Optimal Mel Spectrogram at step {k}', save_path='./visualisations/final_mel.png', dB_scale=False)
            plot_activation(avg, k, save_path='./visualisations/activation.png')
            # plot_waveform(input.detach().squeeze(0), sample_rate=44100, title='Optimal Waveform', save_path='./visualisations/final_waveform.png')
            # if k % 50 == 0:
            #     torch.save(latent_means, f'./visualisations/activation_pattern_step_{k}.pt')

        # Keep highest activation
        if best_activation > act:
            best_activation = act
            best_img = input

    return best_img

"""
Prepare Input from Image
"""
def load_image(path_to_image, device=False):
    tensor_image = get_image(path_to_image)
    tensor_image = normalize(tensor_image)
    tensor_image = tensor_image.unsqueeze(0)
    tensor_image.requires_grad = True
    if device:
        tensor_image = tensor_image.type(torch.cuda.FloatTensor)
    else:
        tensor_image = tensor_image.type(torch.FloatTensor)
    return tensor_image

"""
Prepare Dummy Input
"""
def load_dummy_image(device=False):
    dummy_image = torch.randn(3, 256, 256, requires_grad=True)
    dummy_image = normalize(dummy_image)
    if device:
        tensor_image = tensor_image.type(torch.cuda.FloatTensor)
    else:
        tensor_image = tensor_image.type(torch.FloatTensor)
    dummy_image = dummy_image.unsqueeze(0)
    return dummy_image
