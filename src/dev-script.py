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

for i in range(len(data)):
	data_x = (data[i] - 7.5)/ 7.5
	data_y = np.eye(16)[data[i]]
	inputs.append(data_x.reshape(1, data_x.shape[0], 1))
	labels.append(data_y.reshape(1, data_y.shape[0], 16))

print inputs[0][:, 0:10000, :].shape
print labels[0][:, 100:10100, :].shape

batch_size = 1
global_context_size = 100
bptt_steps = 100
input = tf.placeholder(tf.float32, [batch_size, global_context_size*bptt_steps+global_context_size-1, 1])
label = tf.placeholder(tf.uint8, [batch_size, global_context_size*bptt_steps+global_context_size, 16])
t_model = model.sample_rnn(input, label)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print sess.run(t_model.o, feed_dict={input:inputs[0][:, :10099, :], label:labels[0][:, 100:10100, :]})