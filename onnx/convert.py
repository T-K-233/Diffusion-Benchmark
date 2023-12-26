import sys
sys.path.append("../Diffusion-Benchmark/")

import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

from model import TransformerForDiffusion


device = "cpu"

# # GPT with time embedding and obs cond
# model = TransformerForDiffusion(
#     input_dim=16,
#     output_dim=16,
#     horizon=8,
#     n_obs_steps=4,
#     cond_dim=10,
#     n_layer = 6,
#     n_head = 8,
#     n_emb = 256,
#     time_as_cond=True,
#     obs_as_cond=False,
#     device=device
# )

model = torch.load("model_full.pt")

t = 0
sample = torch.zeros((4, 8, 16), dtype=torch.float32, device=device)
timestep = torch.tensor([t, t, t, t], dtype=torch.float32, device=device)
cond = torch.zeros((4, 4, 10), dtype=torch.float32, device=device)

# (4, 8, 16)
out = model.forward(sample, timestep, cond)

# Export the model
torch.onnx.export(model,               # model being run
                  (sample, timestep, cond),                         # model input (or a tuple for multiple inputs)
                  "model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=17,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ["sample", "timestep", "cond"],   # the model's input names
                  output_names = ["action"], # the model's output names
                  dynamic_axes={
                      "sample" :    {0 : "batch_size"},    # variable length axes
                      "timestep" :  {0 : "batch_size"},
                      "cond" :      {0 : "batch_size"},
                      "action" :    {0 : "batch_size"}})
