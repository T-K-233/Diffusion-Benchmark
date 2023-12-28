import sys
sys.path.append("../Diffusion-Benchmark/")

import os

import numpy as np
import torch
import tensorrt as trt
from cuda import cudart

from inference import sample, timestep, cond


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
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
