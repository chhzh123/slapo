# QAT follows the same steps as PTQ, with the exception of the training loop before you actually convert the model to its quantized version
# https://pytorch.org/blog/quantization-in-practice/#quantization-aware-training-qat

import torch
from torch import nn

backend = "fbgemm"  # running on a x86 CPU. Use "qnnpack" if running on ARM.

m = nn.Sequential(
     nn.Conv2d(2,64,8),
     nn.ReLU(),
     nn.Conv2d(64, 128, 8),
     nn.ReLU()
)
print("Original", m)

"""Fuse"""
torch.quantization.fuse_modules(m, ['0','1'], inplace=True) # fuse first Conv-ReLU pair
torch.quantization.fuse_modules(m, ['2','3'], inplace=True) # fuse second Conv-ReLU pair
print("Fuse", m)

"""Insert stubs"""
m = nn.Sequential(torch.quantization.QuantStub(), 
                  *m, 
                  torch.quantization.DeQuantStub())
print("Stub", m)

"""Prepare"""
m.train()
m.qconfig = torch.quantization.get_default_qconfig(backend)
torch.quantization.prepare_qat(m, inplace=True)
print("Prepare", m)
# print(type(getattr(m, "1")))
# print(type(getattr(m, "1").weight_fake_quant))
# print(type(getattr(m, "1").activation_post_process))
# sys.exit()

"""Training Loop"""
n_epochs = 10
opt = torch.optim.SGD(m.parameters(), lr=0.1)
loss_fn = lambda out, tgt: torch.pow(tgt-out, 2).mean()
for epoch in range(n_epochs):
  x = torch.rand(10,2,24,24)
  out = m(x)
  loss = loss_fn(out, torch.rand_like(out))
  opt.zero_grad()
  loss.backward()
  print("Epoch", epoch, "Loss", loss.item())
  opt.step()

"""Convert"""
m.eval()
torch.quantization.convert(m, inplace=True)
print("Trained", m)
