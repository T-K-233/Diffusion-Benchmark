import torch

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

# inference device
device = "cpu"

sample = torch.rand((1, 16, 12), dtype=torch.float32, device=device)
timestep = torch.rand((1, ), dtype=torch.float32, device=device)
cond = torch.rand((1, 8, 42), dtype=torch.float32, device=device)

result = torch.tensor([-0.347064,  0.67148 , -0.176768, -0.625894, -0.092198, -0.515635,  0.160883, -0.045024,  0.071995,  0.592961,  0.498511,  1.126697])

