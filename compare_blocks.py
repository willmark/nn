import torch
import torch.nn as nn

torch.manual_seed(0)

# Residual and Plain blocks
class ResidualBlock(nn.Module):
    def __init__(self, size): super().__init__(); self.lin = nn.Linear(size, size)
    def forward(self, x): return x + self.lin(x)

class PlainBlock(nn.Module):
    def __init__(self, size): super().__init__(); self.lin = nn.Linear(size, size)
    def forward(self, x): return self.lin(x)

# Create networks
def make_net(block, size=3, depth=10): return nn.Sequential(*[block(size) for _ in range(depth)])

# Shared input
x_res = torch.tensor([[1., 2., 3.]], requires_grad=True)
x_plain = x_res.clone().detach().requires_grad_()

# Models
res_net = make_net(ResidualBlock)
plain_net = make_net(PlainBlock)

# Forward pass with activation tracking
def run(model, x_input):
    acts = []
    x = x_input
    for layer in model:
        x = layer(x)
        acts.append(x)
    y = x.sum(); y.backward()
    return acts, x_input.grad, model[0].lin.weight.grad

# Run both
acts_r, grad_x_r, grad_w_r = run(res_net, x_res)
acts_p, grad_x_p, grad_w_p = run(plain_net, x_plain)

# Results
print("Residual - Final Output:", acts_r[-1].detach().numpy())
print("Residual - Grad Input:", grad_x_r.detach().numpy())
print("Residual - Grad W[0]:", grad_w_r.detach().numpy())

print("\nPlain - Final Output:", acts_p[-1].detach().numpy())
print("Plain - Grad Input:", grad_x_p.detach().numpy())
print("Plain - Grad W[0]:", grad_w_p.detach().numpy())
