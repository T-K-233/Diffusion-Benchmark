import time

import numpy as np
import torch
import onnxruntime
import tensorrt as trt
from cuda import cudart


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

with open("run.log", "w") as f:
    f.write("")


print("Start benchmarking...")

sample_shape = (1, 16, 12)
timestep_shape = (1, )
cond_shape = (1, 8, 42)

N_ITRS = 1000


# ===========================
# CPU PyTorch Native
# ===========================
device = "cpu"
torch_model = torch.load("model_full.pt")

torch_model.eval()
torch_model.to(device)

for i in range(N_ITRS):    
    with torch.no_grad():
        sample = torch.rand(sample_shape, dtype=torch.float32, device=device)
        timestep = torch.rand(timestep_shape, dtype=torch.float32, device=device)
        cond = torch.rand(cond_shape, dtype=torch.float32, device=device)
        out = torch_model.forward(sample, timestep, cond)

start = time.time()

for i in range(N_ITRS):
    with torch.no_grad():
        sample = torch.rand(sample_shape, dtype=torch.float32, device=device)
        timestep = torch.rand(timestep_shape, dtype=torch.float32, device=device)
        cond = torch.rand(cond_shape, dtype=torch.float32, device=device)
        out = torch_model.forward(sample, timestep, cond)

dt = time.time() - start
print("PyTorch Native-CPU - Performance: {0:.3f} / kitr - Rate: {1:.3f} Hz".format(dt, 1000 / dt / 20))
with open("run.log", "a") as f:
    f.write("PyTorch Native-CPU - Performance: {0:.3f} / kitr - Rate: {1:.3f} Hz\n".format(dt, 1000 / dt / 20))


# ===========================
# GPU PyTorch Native
# ===========================
device = "cuda"
torch_model = torch.load("model_full.pt")

torch_model.eval()
torch_model.to(device)
torch_model.time_emb.emb = torch_model.time_emb.emb.to(device)

for i in range(N_ITRS):    
    with torch.no_grad():
        sample = torch.rand(sample_shape, dtype=torch.float32, device=device)
        timestep = torch.rand(timestep_shape, dtype=torch.float32, device=device)
        cond = torch.rand(cond_shape, dtype=torch.float32, device=device)
        out = torch_model.forward(sample, timestep, cond)

start = time.time()

for i in range(N_ITRS):
    with torch.no_grad():
        sample = torch.rand(sample_shape, dtype=torch.float32, device=device)
        timestep = torch.rand(timestep_shape, dtype=torch.float32, device=device)
        cond = torch.rand(cond_shape, dtype=torch.float32, device=device)
        out = torch_model.forward(sample, timestep, cond)

dt = time.time() - start
print("PyTorch Native-GPU - Performance: {0:.3f} / kitr - Rate: {1:.3f} Hz".format(dt, 1000 / dt / 20))
with open("run.log", "a") as f:
    f.write("PyTorch Native-GPU - Performance: {0:.3f} / kitr - Rate: {1:.3f} Hz\n".format(dt, 1000 / dt / 20))


# ===========================
# CPU PyTorch Compile With Inductor
# ===========================
device = "cpu"
torch_model = torch.load("model_full.pt")

torch_model.eval()
torch_model.to(device)

torch_model = torch.compile(torch_model, backend="inductor")

for i in range(N_ITRS):    
    with torch.no_grad():
        sample = torch.rand(sample_shape, dtype=torch.float32, device=device)
        timestep = torch.rand(timestep_shape, dtype=torch.float32, device=device)
        cond = torch.rand(cond_shape, dtype=torch.float32, device=device)
        out = torch_model.forward(sample, timestep, cond)

start = time.time()

for i in range(N_ITRS):
    with torch.no_grad():
        sample = torch.rand(sample_shape, dtype=torch.float32, device=device)
        timestep = torch.rand(timestep_shape, dtype=torch.float32, device=device)
        cond = torch.rand(cond_shape, dtype=torch.float32, device=device)
        out = torch_model.forward(sample, timestep, cond)

dt = time.time() - start
print("PyTorch Inductor Compiled - Performance: {0:.3f} / kitr - Rate: {1:.3f} Hz".format(dt, 1000 / dt / 20))
with open("run.log", "a") as f:
    f.write("PyTorch Inductor Compiled - Performance: {0:.3f} / kitr - Rate: {1:.3f} Hz\n".format(dt, 1000 / dt / 20))


# ===========================
# CPU ONNX Runtime
# ===========================
ort_session = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

for i in range(N_ITRS):
    sample = np.random.random_sample(sample_shape).astype(np.float32)
    timestep = np.random.random_sample(timestep_shape).astype(np.float32)
    cond = np.random.random_sample(cond_shape).astype(np.float32)
    ort_inputs = {
        ort_session.get_inputs()[0].name: sample,
        ort_session.get_inputs()[1].name: timestep,
        ort_session.get_inputs()[2].name: cond,
        }
    ort_outs = ort_session.run(None, ort_inputs)

start = time.time()

for i in range(N_ITRS):
    sample = np.random.random_sample(sample_shape).astype(np.float32)
    timestep = np.random.random_sample(timestep_shape).astype(np.float32)
    cond = np.random.random_sample(cond_shape).astype(np.float32)
    ort_inputs = {
        ort_session.get_inputs()[0].name: sample,
        ort_session.get_inputs()[1].name: timestep,
        ort_session.get_inputs()[2].name: cond,
        }
    ort_outs = ort_session.run(None, ort_inputs)

dt = time.time() - start
print("ONNX-CPU - Performance: {0:.3f} / kitr - Rate: {1:.3f} Hz".format(dt, 1000 / dt / 20))
with open("run.log", "a") as f:
    f.write("ONNX-CPU - Performance: {0:.3f} / kitr - Rate: {1:.3f} Hz\n".format(dt, 1000 / dt / 20))


# ===========================
# GPU ONNX Runtime
# ===========================
ort_session = onnxruntime.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

for i in range(N_ITRS):
    sample = np.random.random_sample(sample_shape).astype(np.float32)
    timestep = np.random.random_sample(timestep_shape).astype(np.float32)
    cond = np.random.random_sample(cond_shape).astype(np.float32)
    ort_inputs = {
        ort_session.get_inputs()[0].name: sample,
        ort_session.get_inputs()[1].name: timestep,
        ort_session.get_inputs()[2].name: cond,
        }
    ort_outs = ort_session.run(None, ort_inputs)


start = time.time()

for i in range(N_ITRS):
    sample = np.random.random_sample(sample_shape).astype(np.float32)
    timestep = np.random.random_sample(timestep_shape).astype(np.float32)
    cond = np.random.random_sample(cond_shape).astype(np.float32)
    ort_inputs = {
        ort_session.get_inputs()[0].name: sample,
        ort_session.get_inputs()[1].name: timestep,
        ort_session.get_inputs()[2].name: cond,
        }
    ort_outs = ort_session.run(None, ort_inputs)

dt = time.time() - start
print("ONNX-GPU - Performance: {0:.3f} / kitr - Rate: {1:.3f} Hz".format(dt, 1000 / dt / 20))
with open("run.log", "a") as f:
    f.write("ONNX-GPU - Performance: {0:.3f} / kitr - Rate: {1:.3f} Hz\n".format(dt, 1000 / dt / 20))


# ===========================
# TensorRT FP32 Runtime
# ===========================
cudart.cudaDeviceSynchronize()
logger = trt.Logger(trt.Logger.WARNING)
with open("model.plan", "rb") as f:
    engineString = f.read()


engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()


for i in range(N_ITRS):
    sample = np.random.random_sample(sample_shape)
    timestep = np.random.random_sample(timestep_shape)
    cond = np.random.random_sample(cond_shape)

    bufferH = []
    bufferH.append(np.ascontiguousarray(sample))
    bufferH.append(np.ascontiguousarray(timestep))
    bufferH.append(np.ascontiguousarray(cond))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    status = context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for b in bufferD:
        cudart.cudaFree(b)

start = time.time()

for i in range(N_ITRS):
    sample = np.random.random_sample(sample_shape)
    timestep = np.random.random_sample(timestep_shape)
    cond = np.random.random_sample(cond_shape)

    bufferH = []
    bufferH.append(np.ascontiguousarray(sample))
    bufferH.append(np.ascontiguousarray(timestep))
    bufferH.append(np.ascontiguousarray(cond))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    status = context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for b in bufferD:
        cudart.cudaFree(b)

dt = time.time() - start
print("TensorRT - Performance: {0:.3f} / kitr - Rate: {1:.3f} Hz".format(dt, 1000 / dt / 20))
with open("run.log", "a") as f:
    f.write("TensorRT - Performance: {0:.3f} / kitr - Rate: {1:.3f} Hz\n".format(dt, 1000 / dt / 20))



