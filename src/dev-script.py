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
	data_x = (data[i] )/ 3.75
	data_y = np.eye(16)[data[i]]
	inputs.append(data_x.reshape(1, data_x.shape[0], 1))
	labels.append(data_y.reshape(1, data_y.shape[0], 16))

batch_size = 1
global_context_size = 100
bptt_steps = 2
n_epochs = 100
clip_iter = 20
input = tf.placeholder(tf.float32, [batch_size, global_context_size*bptt_steps+global_context_size-1, 1])
label = tf.placeholder(tf.uint8, [batch_size, global_context_size*bptt_steps, 16])
t_model = model.sample_rnn(input, label, is_training=False)

optimizer = tf.train.AdadeltaOptimizer(0.01)
global_step = tf.Variable(0)
gradients, v = zip(*optimizer.compute_gradients(t_model.loss))
gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
saver = tf.train.Saver()
if not os.path.exists('./params'):	os.makedirs('./params')


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	z_state = (t_model.initial_state[0].eval(), t_model.initial_state[1].eval())

	for ep in range(n_epochs):
		print 'epoch:', ep
		for i in range(len(inputs)):
			for ci in range(clip_iter):
				print 'Training on clip #', i, '/', len(inputs)
				print 'Iteration:', ci
				current_clip = inputs[i]
				n_bptt_batches = current_clip.shape[1] / (global_context_size * bptt_steps) - 1
				np_state = z_state
				for j in range(n_bptt_batches):
					start_ptr = j*global_context_size*bptt_steps
					end_ptr = (j+1)*global_context_size*bptt_steps + global_context_size - 1
					bptt_batch_x = current_clip[:, start_ptr:end_ptr, :]
					bptt_batch_y = labels[i][:, start_ptr+global_context_size:end_ptr+1, :]
					bptt_batch_loss, acc, np_state, op, out, p1, p2 = sess.run([t_model.loss, t_model.mean_acc, t_model.final_state, optimizer, t_model.outputs, t_model.o1, t_model.o2], feed_dict={input:bptt_batch_x, label:bptt_batch_y,
															t_model.initial_state[0]:np_state[0], t_model.initial_state[1]:np_state[1]})
					print 'clipiter:', ci,', bptt index:', j, ':, loss:', bptt_batch_loss, ', accuracy:', acc*100., '%'
					#print 'memory activations: ', 100. - ((np_state[0]<0).sum() + (np_state[1]<0).sum())/2., '%' #remove
					#print 'bincount: ', np.bincount(out) # remove
					#print p1 # remove
					#print p2 # remove
				save_path = saver.save(sess, "./params/last_model.ckpt")
				print("Model saved in file: %s\n" % save_path)
