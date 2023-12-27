import sys
sys.path.append("../Diffusion-Benchmark/")

import os
from datetime import datetime as dt
from glob import glob
import time

import numpy as np
import tensorrt as trt
import torch
from torch import nn
import torch.nn.functional as F
from cuda import cudart
from torch.autograd import Variable

from model import TransformerForDiffusion

np.random.seed(31193)
torch.manual_seed(97)
torch.cuda.manual_seed_all(97)
torch.backends.cudnn.deterministic = True


onnxFile = "./model.onnx"
trtFile = "./model.plan"



# for FP16 mode
bUseFP16Mode = False
# for INT8 model
bUseINT8Mode = False
nCalibration = 1
cacheFile = "./int8.cache"

# os.system("rm -rf ./*.onnx ./*.plan ./*.cache")
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()



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
    device="cpu"
)

n_infer = 1000

timestep = torch.tensor([0.]*4, dtype=torch.float32)
sample = torch.zeros((4, 8, 16), dtype=torch.float32)
cond = torch.zeros((4, 4, 10), dtype=torch.float32)

xTest = (sample, timestep, cond)

model.load_state_dict(torch.load("model.pt"))
#model.eval()

# with torch.no_grad():
#     for i in range(n_infer):
#         y_ = model.forward(sample, timestep, cond)

#     start_time = time.time()
#     for i in range(n_infer):
#         y_ = model.forward(sample, timestep, cond)
#     # 1.6s
#     print("time taken torch:", time.time() - start_time)
#     #breakpoint()


print("Succeeded building model in pyTorch!")

# Export model as ONNX file ----------------------------------------------------
torch.onnx.export(
    model, 
    xTest,
    #torch.randn(1, 1, nHeight, nWidth, device="cuda"), 
    onnxFile, 
    input_names=["sample", "timestep", "cond"], 
    output_names=["action"], 
    do_constant_folding=True, 
    verbose=True, 
    keep_initializers_as_inputs=True, 
    opset_version=17, 
    # dynamic_axes={
    #     'sample' : {0 : 'batch_size'},    # variable length axes
    #     'timestep' : {0 : 'batch_size'},
    #     'cond' : {0 : 'batch_size'},
    #     'action' : {0 : 'batch_size'}
    # }
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
opt_shape = [4]
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
# context.set_input_shape(lTensorName[0], sample.shape)
# context.set_input_shape(lTensorName[1], [4])
# context.set_input_shape(lTensorName[2], cond.shape)
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])



start_time = time.time()

# lTensorName: ['sample', 'timestep', 'cond', 'action']
# nIO: 4
# nInput: 3

for i in range(n_infer):
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
