import numpy as np
import matplotlib.pyplot as plt
# https://www.youtube.com/watch?v=l-Fe9Ekxxj4

w = 1.0
x_data = [1, 2, 3]
y_data = [2, 4, 6]

#em forma geral as redes neuroanis fazem isto:
def fordward(x):
	return x*w

def loss(y, y_hat):
	return (y_hat-y)*(y_hat-y)

w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
	print('w=', w)
	val_sum = 0
	for x_val, y_val in zip(x_data, y_data):
		y_hat = fordward(x_val)
		l = loss(y_val, y_hat)
		val_sum +=l
		print('\t', x_val, y_val, y_hat, l)
	print('MSE=', val_sum/3)
	w_list.append(w)
	mse_list.append(val_sum/3)


plt.plot(w_list, mse_list)
plt.ylabel('loss')
plt.xlabel('w')
plt.show()


