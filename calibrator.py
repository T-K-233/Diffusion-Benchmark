#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

import torch
import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart


class MyCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, nCalibration, cacheFile="./int8.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.timestep = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        self.sample = torch.zeros((4, 8, 16), dtype=torch.float32)
        self.cond = torch.zeros((4, 4, 10), dtype=torch.float32)

        self.nCalibration = nCalibration
        self.shape = (4, 8, 16)
        size = trt.volume(self.timestep.shape) + trt.volume(self.sample.shape) + trt.volume(self.cond.shape)
        self.buffeSize = size * trt.float32.itemsize
        self.cacheFile = cacheFile
        # _, self.dIn = cudart.cudaMalloc(self.buffeSize)

        bufferH = []
        bufferH.append(np.ascontiguousarray(self.sample.cpu().numpy()))
        bufferH.append(np.ascontiguousarray(self.timestep.cpu().numpy()))
        bufferH.append(np.ascontiguousarray(self.cond.cpu().numpy()))
        
        self.dIn = []
        for i in range(3):
            self.dIn.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        
        self.oneBatch = self.batchGenerator()

        self.cnt = 0

        print(int(self.dIn[0]))

    def __del__(self):
        for b in self.dIn:
            cudart.cudaFree(b)

    def batchGenerator(self):
        for i in range(self.nCalibration):
            yield (
                np.ascontiguousarray(self.sample.cpu().numpy()),
                np.ascontiguousarray(self.timestep.cpu().numpy()),
                np.ascontiguousarray(self.cond.cpu().numpy()),
            )

    def get_batch_size(self):  # necessary API
        return 4

    def get_batch(self, nameList=None, inputNodeName=None):  # necessary API
        print("> calibration %d" % self.cnt)
        self.cnt += 1

        if self.cnt > 2:
            return None
        return [int(self.dIn[0]), int(self.dIn[1]), int(self.dIn[2])]

    def read_calibration_cache(self):  # necessary API
        if os.path.exists(self.cacheFile):
            print("Succeed finding cahce file: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return None

    def write_calibration_cache(self, cache):  # necessary API
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")
        return

if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    m = MyCalibrator("../../00-MNISTData/test/", 5, (1, 1, 28, 28), "./int8.cache")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
