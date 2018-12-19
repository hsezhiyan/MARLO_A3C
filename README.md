# Overview

This is a A3C (Actor Critic) method for solving the Find-the-Goal minitask in the Minecraft (MARLO) environment.

# Technologies Used
  - Chainerrl
  - Pytorch
  - Autoencoding

# How it works

The core implementation involves A3C, which combines a policy and value estimation into one architecture. We used Chainer's A3C implentation, with slight changes to incorporate our autoencoder encoder model, discussed below.

# Autoencoder

We used a (nonvariational) autoencoder to compress the frame inputs by 250 times. The structure for this autoencoder is written in the file AutoEncoderModels.py in the path /chainerrl_autoencoder/experimetes_ae/. It involves 10 residual blocks and 2 deconvolution layers. chainerrl_autoencoder is essentially the original implementation of chainerrl, with small dimenionsional changes to account for the autoencoder.

The following are compressed images of the Minecraft environment by the autoencoder:

 <br><br> <img src="/img/ae_test_images.png" height="500" width="500" alt="Autoencoder"/>

# Results

Results for training A3C model with Autoencoder on Find-the-Goal minitask.

 <br><br> <img src="/img/Reward_curve.png" height="500" width="500" alt="Reward curve"/>