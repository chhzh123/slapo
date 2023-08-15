<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# LLaMA Benchmark

Please follow the following instructions to set up the environment.

```bash
# Install Slapo
git clone https://github.com/chhzh123/slapo.git
cd slapo
git checkout autoshard
pip3 install -e ".[dev]"

# Install PyTorch
# Please refer to https://pytorch.org/ for correct OS and CUDA version
pip3 install torch==2.0.1 torchvision

# Install HuggingFace Transformers v4.28.1 (the upstream one may have compatibility issues with FX tracers)
git clone https://github.com/chhzh123/transformers.git
cd transformers
git checkout slapo
pip3 install -e .
# Used for transformer kernels
sudo apt-get install ninja-build

# Install DeepSpeed (An in-house version customized by Zhen, which maximizes the performance of pipeline parallelism with MiCS-pipe)
git clone https://github.com/dmlc/DeepSpeed.git
cd DeepSpeed
git checkout pipe
pip3 install -e .

# Install other dependencies
pip3 install datasets matplotlib tabulate networkx triton pybind11 sentencepiece
```