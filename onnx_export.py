
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

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

model = torch.compile(model)


timestep = torch.tensor([0], dtype=torch.float32, device=device)
sample = torch.zeros((4, 8, 16), dtype=torch.float32, device=device)
cond = torch.zeros((4, 4, 10), dtype=torch.float32, device=device)


out = model.forward(sample, timestep, cond)


# Export the model
torch.onnx.export(model,               # model being run
                  (timestep, sample, cond),                         # model input (or a tuple for multiple inputs)
                  "transformer_diff.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})



