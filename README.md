
# Langevin Dynamics Sampling of Prompt Distribution for Vision-Language Model

## Background
This repository uses Langevin Dynamics in optimisation where each step of the optimisation update resembles the sampled prompt parameters from the underlying prompt distribution. By saving the sampled prompt along the Langevin Dynamics Sampling trajectory, we then later used them to conduct hard ensemble voting in test time prediction. The intuitive idea of Langevin Dynamics Sampling versus the variational approach is shown in the figure below:

<figure align="center">
  <img src="figures/variational_vs_sampling.png", style="width:450px;">
  <br> Fig.1: The distribution in green is the target distribution to estimate. The orange distribution in the left fits a Gaussian function on the target green distribution, while the orange dots in the right represent a direct sampling from the target green distribution where those dots reside in the high probability region of the target distribution.
</figure>


To better show how the sampling is performed in the Langevin Dynamics, consider a stationary distribution in Fig. 2 which is unknown in its analytical expression to the Langevin Dynamics particle, where the darker colour represent a higher density region, and the orange dot represent the footprint of the Langevin Dynamics particle's movement. As the iteration progress from 1 to 240, we can see the orange footprints visit the high density region more, hence recovering the underlying unknown blue distribution.


<p align="center">
  <img src="figures/LD_iter0.png" width="150" />
  <img src="figures/LD_iter40.png" width="150" /> 
  <img src="figures/LD_iter240.png" width="150" />
  <br>
    Fig 2: Langevin Dynamics particle movement.
</p>


## Setup

1. Download the official cifar dataset from [Here](https://www.cs.toronto.edu/~kriz/cifar.html) and put is under the folder ``datasets'' in the project directory as 
  ```
  COOP_Langevin_Public
  ├── datasets
  │   ├── cifar-10-python
  │   ├── cifar-100-python
  │...
  ```


2. Create the conda env and clone this repository

    ```
    # Clone this repo
    https://github.com/Walter-pixel/COOP_Langevin_Public.git
    cd Environment/env

    # Create a conda environment
    conda create -y -n LD_env python=3.9.16

    # Activate the environment
    conda activate LD_env

    # Install torch and torchvision
    # Please refer to https://pytorch.org/ if you need a different cuda version
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia


    # Install dependencies
    cd ../../
    pip install -r requirements.txt

    ```
3. item 3

## Result on Cifar10-LT


| Method       | Learn   | Imb=200 | Imb=100 | Imb=50 | Imb=10 |
|--------------|---------|---------|---------|--------|--------|
| Variational  | Prompt  | 77.06   | 78.50   | 79.03  | 79.46  |
| LD 1 model   | Prompt  | 75.25   | 76.15   | 78.33  | 78.70  |
| LD 80 models | Prompt  | 76.34   | 77.86   | 78.60  | 79.56  |






