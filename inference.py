import torch

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

# inference device
device = "cpu"

sample = torch.rand((1, 16, 12), dtype=torch.float32, device=device)
timestep = torch.rand((1, ), dtype=torch.float32, device=device)
cond = torch.rand((1, 8, 42), dtype=torch.float32, device=device)

