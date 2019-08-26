import torch
from torch.nn import Linear

w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

torch.manual_seed(1)
model = Linear(in_features=1, out_features=1)

print(model.bias, model.weight)

def forward(x):
	y = w*x + b
	return y
x = torch.tensor([2])
x = torch.tensor([[2.0], [3.3]])

forward(x)
print(model(x))