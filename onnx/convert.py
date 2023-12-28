import sys
sys.path.append("../Diffusion-Benchmark/")

import torch.onnx

from inference import sample, timestep, cond


onnx_file = "./model.onnx"

model = torch.load("model_full.pt")


# Export model as ONNX file ----------------------------------------------------
torch.onnx.export(
    model, 
    (sample, timestep, cond),
    onnx_file, 
    input_names=["sample", "timestep", "cond"], 
    output_names=["action"], 
    do_constant_folding=True, 
    verbose=True, 
    keep_initializers_as_inputs=True, 
    opset_version=17, 
    dynamic_axes={}
    )

print("Succeeded converting model into ONNX!")
