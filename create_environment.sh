#!/bin/bash

git submodule update --init --recursive --jobs 0


source $(conda info --base)/etc/profile.d/conda.sh

conda update -n base -c defaults conda

conda create -y -n trips python=3.9.7
conda activate trips

conda install -y cuda  -c nvidia/label/cuda-11.8.0
conda install -y  mkl=2022.0.1 mkl-include=2022.0.1
conda install -y cmake=3.26.4

conda install -y -c conda-forge glog=0.5.0 gflags=2.2.2 protobuf=3.13.0.1 freeimage=3.17 tensorboard=2.8.0

# ./install_pytorch_precompiled.sh
