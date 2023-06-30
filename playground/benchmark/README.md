# Slapo Inference Benchmark

Please clone the following projects to the same directory:
```bash
# Install Slapo
# Make sure you have already installed Pytorch 2.0 and Huggingface Transformers
git clone https://github.com/chhzh123/slapo.git
cd slapo
git checkout autoshard
pip install -e ".[dev]"

cd playground/benchmark/deepspeed
mkdir build && cd build
cmake ..
make

# Install DeepSpeed
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
pip install -e .

# Install FasterTransformer
git clone https://github.com/chhzh123/FasterTransformer.git
cd FasterTransformer
git checkout torch
cd torch
mkdir build && cd build
cmake ..
make

# Install ByteTransformer
git clone https://github.com/chhzh123/ByteTransformer.git
cd ByteTransformer
git checkout pytorch
cd torch
mkdir build && cd build
cmake ..
make
```

Run the benchmarking program:
```bash
# Run Slapo
torchrun --nproc_per_node 8 bench_slapo_inference.py --name bert

# Run DeepSpeed
deepspeed --num_gpus 8 bench_ds_inference.py --name bert
```
