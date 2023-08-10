#!/bin/bash

conda create -n pt20 python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate pt20
sudo apt-get install ninja-build
pip3 install torch==2.0.1 torchvision datasets matplotlib tabulate networkx triton pybind11 sentencepiece
cd /fsx/zhzhn/slapo
pip install -e .
cd /fsx/zhzhn/DeepSpeed
pip install .
cd /fsx/zhzhn/transformers
pip install -e .