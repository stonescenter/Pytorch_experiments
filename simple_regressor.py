import torch
import torch.nn as nn
from torch.autograd import Variable
# https://www.youtube.com/watch?v=l-Fe9Ekxxj4

class Model(nn.Module):

	def __init__(self, input_size, output_size): 
		super(Model, self).__init__()

		self.input_size = input_size
		self.output_size = output_size

		
		self.linear = nn.Linear(self.input_size, self.output_size)


	def forward(self, x):

		y_pred = self.linear(x)

		return y_pred

	def train(self, nada):
		pass


# inializamos aleatoriamente
w = Variable(torch.Tensor([1.0]), requires_grad=True)

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

# em forma geral as redes neuroanis fazem isto:
# def fordward(x):
# 	return x*w

# def loss(x, y):
# 	y_pred = fordward(x)
# 	return (y_pred-y)*(y_pred-y)


# for epoch in range(100):
# 	for x_val, y_val in zip(x_data, y_data):
# 		l = loss(x_val, y_val)
# 		l.backward()
# 		print('grad:', x_val, y_val, w.grad.data[0])
# 		w.data = w.data - 0.01 * w.grad.data
# 		w.grad.data.zero_()

# 	print('progress:', epoch, l.data[0])

input_size = 4 
linear = Model(1, 1)
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

print(linear)

for epoch in range(400):

	optimizer.zero_grad()
	y_pred = linear(x_data)
	loss = criterion(y_pred, y_data)
	print('progress:', epoch, loss.item())
	loss.backward()
	optimizer.step()

