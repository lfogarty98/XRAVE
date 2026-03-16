
## Broad Context & Motivation (RAVE)
- Despite its age, RAVE (2021) has prevailed as the most popular tool for real-time NAS-based musical interaction (NAS: neural audio synthesis)
- It used by artists primarily in two ways: timbre transfer and unconditional generation
    - *Timbre transfer*: an out-of-distribution input audio signal conditions the output of the decoder
    - *Unconditional generation*: decoder-only generation via a pretrained "prior network"
- In both cases, the generation can be controlled by manipulating a set of latent synthesis parameters - one of the most compelling aspects of NAS. In the case of RAVE, these parameters are the PCA axes of the (typically 128-dimensional) latent space
- An open question is: how well defined/meaningful are these manipulations?

## Problem Formulation
- Training a RAVE model involves two stages. The first is termed *representation learning*, in which the encoder and decoder are trained jointly using a VAE objective. It is during this phase that the latent representation of the dataset is learnt.
- To improve its synthesised audio quality, this is followed by a second adversarial training stage, in which the encoder is frozen and the decoder is further optimised with respect to a new GAN-based objective
- An observation here is that this GAN phase can lead to the decoder mapping diverging further from the encoder mapping. This may have the consequence of output generations not aligning meaningfully with how the input was encoded. For instance:
    - Timbre transfer case: only high-level attributes like fundamental frequency and loudness are meaningfully carried over into the decoded signal, with finer timbral structure being ignored
    - Unconditional case: latent directions that were semantically organised by the encoder may no longer correspond to perceptually coherent axes in the decoder's output space
- Thesis questions: 
    - Does the adversarial training phase in RAVE lead to less meaningful latent space manipulation? 
    - If so, how? Can we quantifiably illustrate this?

### Central Claim
**The adversarial fine-tuning phase in RAVE systematically degrades the alignment between the encoder's latent organisation and the decoder's generative mapping, reducing the musical meaningfulness of latent space manipulation.**

## Evaluation Framework
- Our goal is to quantify this divergence by comparing offsets of latent synthesis parameters (in the PCA subspace) both from the encoder and decoder side. That is, given PCA latent z_k
    - Encoder side: synthesise an optimal input snippet via ActMax which encodes to this z_k
    - Decoder side: decode z_k to obtain output generation
- We evaluate these spectra empirically as follows:
    - Pick values corresponding to discrete positions along latent PCA axes, with other dimensions at 0, obtaining an array of target z_k
        - _Note_: a more exhaustive approach would be to do a grid search across the PCA space
    - For each z_k, synthesise both ActMax’d (encoder-side) and decoded spectra
    - For each pair of spectra for one z_k: compute metrics
        - _TODO_: define metrics
    - Do the above for (two) models trained on two different datasets, one percussive and one melodic, and compare the results for pre- and post-GAN phase
        - Goal: show that divergence problem is exacerbated by using GAN phase
            - Note: divergence may already be present when inspecting extreme values of latents even without GAN phase. Which can still be worth highlighting and discussing
        - Bonus: *how* does the GAN phase deteriorate the coherency with encoder?
            - Show evolution of metrics over time in GAN phase somehow?

## Implications
- Based on our findings, we discuss implications for artists using RAVE for musical interaction
    - For more coherent encoder-decoder mappings, perhaps forgo the GAN phase when training?
    - Or should the GAN phase also optimise the encoder?