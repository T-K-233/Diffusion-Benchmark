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

## 1070

native (cuda:0): 5.5 s / kitr

native (cpu): 10.5 s / kitr


## 2080

native (cuda:0): 4.2 s / kitr

native (cpu): 4.87 s / kitr

