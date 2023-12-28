import sys
sys.path.append("../Diffusion-Benchmark/")
import time

import torch

from model import TransformerForDiffusion


# inference device
device = "cuda"

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

# GPT with time embedding and obs cond
model = TransformerForDiffusion(device = device)

# torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))

# WARNING: need to set to evaluation moded before saving as final model
model.eval()
torch.save(model, "model_full.pt")

# model = torch.compile(model)


sample = torch.rand((1, 16, 12), dtype=torch.float32, device=device)
timestep = torch.rand((1, ), dtype=torch.float32, device=device)
cond = torch.rand((1, 8, 42), dtype=torch.float32, device=device)

# (1, 16, 12)
out = model.forward(sample, timestep, cond)


# print(out[0][0])
# torch.tensor([-0.3059,  0.7019,  0.3243, -1.0098, -0.6460, -0.4627, -0.7402, -0.3466,  0.2289, -0.2170,  0.8855,  0.8265])


# warmup
for i in range(1000):    
    with torch.no_grad():
        out = model.forward(sample, timestep, cond)

start = time.time()

for i in range(1000):
    with torch.no_grad():
        out = model.forward(sample, timestep, cond)

print(time.time() - start)


