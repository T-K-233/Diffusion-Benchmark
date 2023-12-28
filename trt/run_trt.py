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


onnxFile = "./model.onnx"
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



model = torch.load("model_full.pt")
n_infer = 10000

model.load_state_dict(torch.load("model.pt"))

print("Succeeded building model in pyTorch!")

# Export model as ONNX file ----------------------------------------------------
torch.onnx.export(
    model, 
    (sample, timestep, cond),
    onnxFile, 
    input_names=["sample", "timestep", "cond"], 
    output_names=["action"], 
    do_constant_folding=True, 
    verbose=True, 
    keep_initializers_as_inputs=True, 
    opset_version=17, 
    dynamic_axes={}
    )
print("Succeeded converting model into ONNX!")

# Parse network, rebuild network and do inference in TensorRT ------------------

#logger = trt.Logger(trt.Logger.VERBOSE)
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
if bUseFP16Mode:
    config.set_flag(trt.BuilderFlag.FP16)
if bUseINT8Mode:
    import calibrator
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator.MyCalibrator(nCalibration)

parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxFile):
    print("Failed finding ONNX file!")
    exit()
print("Succeeded finding ONNX file!")
with open(onnxFile, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

inputTensor_sample = network.get_input(0)
inputTensor_time = network.get_input(1)
inputTensor_cond = network.get_input(2)
opt_shape = [sample.shape[0], sample.shape[1], sample.shape[2]]
profile.set_shape(inputTensor_sample.name, opt_shape, opt_shape, opt_shape)
opt_shape = [timestep.shape[0]]
profile.set_shape(inputTensor_time.name, opt_shape, opt_shape, opt_shape)
opt_shape = [cond.shape[0], cond.shape[1], cond.shape[2]]
profile.set_shape(inputTensor_cond.name, opt_shape, opt_shape, opt_shape)
config.add_optimization_profile(profile)

#network.unmark_output(network.get_output(0))  # remove output tensor "y"
engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, "wb") as f:
    f.write(engineString)

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

for i in range(n_infer):

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
