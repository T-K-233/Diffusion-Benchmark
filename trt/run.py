import sys
sys.path.append("../Diffusion-Benchmark/")

import os
from datetime import datetime as dt
from glob import glob
import time

import numpy as np
import torch
import tensorrt as trt
from cuda import cudart

from model import TransformerForDiffusion
from inference import sample, timestep, cond, device, result


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True


trtFile = "./model.plan"


# for FP16 mode
bUseFP16Mode = False
# for INT8 model
bUseINT8Mode = True
nCalibration = 1
cacheFile = "./int8.cache"

# os.system("rm -rf ./*.onnx ./*.plan ./*.cache")
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()


#logger = trt.Logger(trt.Logger.VERBOSE)
logger = trt.Logger(trt.Logger.WARNING)
with open(trtFile, "rb") as f:
    engineString = f.read()

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()

for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])



start_time = time.time()

# lTensorName: ['sample', 'timestep', 'cond', 'action']
# nIO: 4
# nInput: 3

for i in range(1000):

    sample = torch.rand((1, 16, 12), dtype=torch.float32, device=device)
    timestep = torch.rand((1, ), dtype=torch.float32, device=device)
    cond = torch.rand((1, 8, 42), dtype=torch.float32, device=device)

    bufferH = []
    bufferH.append(np.ascontiguousarray(sample.cpu().numpy()))
    bufferH.append(np.ascontiguousarray(timestep.cpu().numpy()))
    bufferH.append(np.ascontiguousarray(cond.cpu().numpy()))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])


    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    res = context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    # for i in range(nIO):
    #     print(lTensorName[i])
    #     print(bufferH[i])
    # print(res)

    for b in bufferD:
        cudart.cudaFree(b)

# 0.3s
print("time taken cuda:", time.time() - start_time)

print("Succeeded running model in TensorRT!")
