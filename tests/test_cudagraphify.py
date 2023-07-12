import copy

import torch
from torch import nn
import slapo


def test_decompose():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            return x

    mod = Model().to(torch.float16).cuda()
    sch = slapo.create_schedule(copy.deepcopy(mod))

    sch["linear"].decompose()
    sch.trace(flatten=True)
    inp = torch.randn((32, 10), dtype=torch.float16, device="cuda", requires_grad=False)
    sch.cudagraphify(example_inputs=[inp])

    sch_model, _ = slapo.build(sch, init_weights=False, dtype=torch.float16)
    print(sch_model)

    out = sch_model(inp)
    out_ref = mod(inp)
    torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_decompose()
