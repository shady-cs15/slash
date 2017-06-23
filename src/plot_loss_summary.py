import matplotlib.pyplot as plt

def f(x, y_list):
	#plt.figure()
	plt.plot(x, y[0], 'ro-', label='train', linewidth=2)
	plt.plot(x, y[1], 'go-', label='valid', linewidth=2)
	plt.xlabel('iters')
	plt.legend()
	plt.grid(True)
	plt.savefig('loss.png')


import numpy as np 
file_ = np.load('loss_summary.npy')
x = file_[0]
y = [file_[1], file_[2]]

f(x, y)