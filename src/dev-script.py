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

batch_size = 1
global_context_size = 100
bptt_steps = 10
n_epochs = 200

input = tf.placeholder(tf.float32, [batch_size, global_context_size*bptt_steps+global_context_size-1, 1])
label = tf.placeholder(tf.uint8, [batch_size, global_context_size*bptt_steps, 16])
t_model = model.sample_rnn(input, label, is_training=True)
optimizer = tf.train.AdamOptimizer().minimize(t_model.loss)
saver = tf.train.Saver()
if not os.path.exists('./params'):	os.makedirs('./params')


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	z_state = (t_model.initial_state[0].eval(), t_model.initial_state[1].eval())

	for ep in range(n_epochs):
		print 'epoch:', ep
		for i in range(len(inputs)):
			print 'Training on clip #', i, '/', len(inputs)
			current_clip = inputs[i]
			n_bptt_batches = current_clip.shape[1] / (global_context_size * bptt_steps) - 1
			np_state = z_state
			for j in range(n_bptt_batches):
				start_ptr = j*global_context_size*bptt_steps
				end_ptr = (j+1)*global_context_size*bptt_steps + global_context_size - 1
				bptt_batch_x = current_clip[:, start_ptr:end_ptr, :]
				bptt_batch_y = labels[i][:, start_ptr+global_context_size:end_ptr+1, :]
				bptt_batch_loss, np_state, op = sess.run([t_model.loss, t_model.final_state, optimizer], feed_dict={input:bptt_batch_x, label:bptt_batch_y, 
														t_model.initial_state[0]:np_state[0], t_model.initial_state[1]:np_state[1]})
				print 'loss at bptt index', j, ':', bptt_batch_loss	
			save_path = saver.save(sess, "./params/last_model.ckpt")
			print("Model saved in file: %s\n" % save_path)
	
