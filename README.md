# Diffusion-Benchmark

## Test condition:

```python

model.eval()

# warmup
for i in range(1000):    
    with torch.no_grad():
        out = model.forward()

# timed
for i in range(1000):
    with torch.no_grad():
        out = model.forward()
```

## Minimal Requirement

20 diffuse steps, 30 Hz

maximum

(1 / 30) / 20 * 1000 = 1.66 s / kitr

## 1070

native (cuda:0): 5.5 s / kitr

native (cpu): 10.5 s / kitr

onnx (cpu): 7.7 s / kitr

trt (cuda:0): 2.2 s / kitr

trt - fp16 (cuda:0): 2.2 s / kitr

trt - int8 (cuda:0): 1.2 s / kitr


## 2080

native (cuda:0): 4.2 s / kitr

native (cpu): 4.87 s / kitr

trt (cuda:0): 0.6 s / kitr

trt - fp16 (cuda:0): 0.5 s / kitr

trt - int8 (cuda:0): 0.5 s / kitr


# TensorRT Convertion Flow


