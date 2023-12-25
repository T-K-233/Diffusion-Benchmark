import time

import torch

from model import TransformerForDiffusion


device = "cpu"

# GPT with time embedding and obs cond
model = TransformerForDiffusion(
    input_dim=16,
    output_dim=16,
    horizon=8,
    n_obs_steps=4,
    cond_dim=10,
    n_layer = 6,
    n_head = 8,
    n_emb = 256,
    time_as_cond=True,
    obs_as_cond=False,
    device=device
)

#torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
model.eval()

# model = torch.compile(model)


timestep = torch.tensor([0], dtype=torch.float32, device=device)
sample = torch.zeros((4, 8, 16), dtype=torch.float32, device=device)
cond = torch.zeros((4, 4, 10), dtype=torch.float32, device=device)


out = model.forward(sample, timestep, cond)

#print(out[0][0])

print(out[0][0] - torch.tensor([ 0.6322, -0.0824,  0.0180,  0.4618,  0.1118, -0.0925,  0.5560, -0.4299,
        -0.1944,  0.4372,  0.6859,  0.0588, -0.8347,  0.8633,  0.0564, -0.1565]).to(device) < 0.001)


# warmup
for i in range(1000):    
    with torch.no_grad():
        out = model.forward(sample, timestep, cond)

start = time.time()

for i in range(1000):
    with torch.no_grad():
        out = model.forward(sample, timestep, cond)

print(time.time() - start)


