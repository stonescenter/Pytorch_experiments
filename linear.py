import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#%matplotlib inline

# create a data
X = torch.randn(100,1)*10
y = X + 3*torch.randn(100,1)

plt.plot(X.numpy(), y.numpy(), 'o')

class LR(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.linear = nn.Linear(input_size, output_size)

	def forward(self, x):
		y = self.linear(x)
		return y

torch.manual_seed(1)
model = LR(1,1)	

[w, b] = model.parameters()

w1 = w[0][0].item()
b1 = b[0].item()


def get_params():
	return (w[0][0].item(), b[0].item())


def plot_fit(title):
	plt.title = title
	w1, b1 = get_params()

	x1 = np.array([-30,30])
	y1 = w1*x1 + b1
	plt.plot(x1, y1, 'r')
	plt.scatter(X,y)
	plt.show()

print(model)
print(w1, b1)
plot_fit('init')


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

epochs = 400
losses = []

print('init...')
for i in range(epochs):
	y_pred = model.forward(X)
	loss = criterion(y_pred, y)
	print("epoch:", i, " loss:", loss.item())
	losses.append(loss)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

print(model)	

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')