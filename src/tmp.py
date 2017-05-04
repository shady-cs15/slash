import os
import numpy as np
import model
import tensorflow as tf

'''
	Normalise inputs and labels
	from .npy files, inputs[i] = ith clip
	labels[i] = one hot encoding of ith clip
'''

data_files = os.listdir('../data/')
data = [np.load('../data/'+ data_file)	for data_file in data_files]
inputs = []
labels = []
batch_size = 64

for i in range(1):#len(data)):
	#data[i] = data[i][:10]
	data_x = (data[i] - 7.5)/ 3.75
	data_y = np.eye(16)[data[i]]
	anchors = np.random.randint(data_x.shape[0]*.6, size=batch_size-1)
	X = (data_x.reshape(1, data_x.shape[0], 1), )
	Y = (data_y.reshape(1, data_y.shape[0], 16), )
	M = (np.ones([1, data_y.shape[0], 16]), )
	for anchor in anchors:
		x = np.append(data_x[anchor:], np.zeros(anchor)).reshape(1, data_x.shape[0], 1)
		y = np.append(data_y[anchor:], np.zeros([anchor, 16])).reshape(1, data_y.shape[0], 16)
		m = np.concatenate((np.ones([1, data_y.shape[0]-anchor, 16]), np.zeros([1, anchor, 16])), axis=1)
		X+= (x, )
		Y+= (y, )
		M+= (m, )
	X = np.concatenate(X)#.reshape([64, data_x.shape[0], 1])
	Y = np.concatenate(Y)#.reshape([64, data_y.shape[0], 16])
	M = np.concatenate(M)
	print X.shape, Y.shape, M.shape
	#print X, Y, M
	np.save('dummyX.npy', X)
	np.save('dummyY.npy', Y)
	np.save('dummyM.npy', M)
	inputs.append(X)
	labels.append(Y)
