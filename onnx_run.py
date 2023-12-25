import numpy as np
import torch
import onnx
import onnxruntime


torch_model = torch.load("model_full.pt")


onnx_model = onnx.load("transformer_diff.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("transformer_diff.onnx", providers=["CUDAExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


timestep = torch.tensor([0], dtype=torch.float32)
sample = torch.zeros((4, 8, 16), dtype=torch.float32)
cond = torch.zeros((4, 4, 10), dtype=torch.float32)


torch_out = torch_model.forward(sample, timestep, cond)

# compute ONNX Runtime output prediction
ort_inputs = {
    ort_session.get_inputs()[0].name: to_numpy(sample),
    ort_session.get_inputs()[1].name: to_numpy(timestep),
    ort_session.get_inputs()[2].name: to_numpy(cond),
    }
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")