import sys
sys.path.append("../Diffusion-Benchmark/")
import time

import numpy as np
import torch
import onnx
import onnxruntime

from inference import sample, timestep, cond, result


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


torch_model = torch.load("model_full.pt")


onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

torch_out = torch_model.forward(sample, timestep, cond)

start_time = time.time()
for i in range(1000):
    # compute ONNX Runtime output prediction
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(sample),
        ort_session.get_inputs()[1].name: to_numpy(timestep),
        ort_session.get_inputs()[2].name: to_numpy(cond),
        }
    ort_outs = ort_session.run(None, ort_inputs)

print("Time taken:", time.time() - start_time)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(result), ort_outs[0][0, 0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
