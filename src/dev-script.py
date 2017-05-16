import os
import numpy as np
import model
import tensorflow as tf
import random

files = os.listdir('../tmp/')
files = [(('../tmp/'+files[i+1]), ('../tmp/'+files[i]))  for i in range(0, len(files), 2)]
random.seed(0)
random.shuffle(files)

inputs = [np.load(files[i][0]) for i in range(len(files)-5)]
masks = [np.load(files[i][1]) for i in range(len(files)-5)]

vinputs = [np.load(files[i][0]) for i in range(len(files)-5, len(files))]
vmasks = [np.load(files[i][1]) for i in range(len(files)-5, len(files))]

batch_size = 64
global_context_size = 100
bptt_steps = 2
n_epochs = 200
clip_iter = 1

input = tf.placeholder(tf.float32, [batch_size, global_context_size*bptt_steps+global_context_size-1, 1])
tf_masks = tf.placeholder(tf.float32, [batch_size, global_context_size*bptt_steps, 1])
tf_inputs = (input - 7.5)/3.75
tf_outputs = tf.placeholder(tf.uint8, [batch_size, global_context_size*bptt_steps, 1])
tf_labels = tf_masks*tf.reshape(tf.one_hot(tf_outputs, depth=16), [batch_size, global_context_size*bptt_steps, 16])
t_model = model.sample_rnn(tf_inputs, tf_labels, tf_masks, batch_size=batch_size, is_training=True)

optimizer = tf.train.AdamOptimizer(0.01)
global_step = tf.Variable(0)
gradients, v = zip(*optimizer.compute_gradients(t_model.loss))
gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
saver = tf.train.Saver()
if not os.path.exists('./params'):	os.makedirs('./params')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	if os.path.exists('./params/last_model.ckpt.meta'):
		saver.restore(sess, './params/last_model.ckpt')
		print 'model restored from last checkpoint..'
	z_state = (t_model.initial_state[0].eval(), t_model.initial_state[1].eval())

	for ep in range(n_epochs):
		print 'epoch:', ep
		for i in range(len(inputs)):
			for ci in range(clip_iter):
				print 'Training on clip #', i, '/', len(inputs)
				current_clip = inputs[i]
				current_mask = masks[i]
				n_bptt_batches = current_clip.shape[1] / (global_context_size * bptt_steps) - 1
				np_state = z_state
				for j in range(n_bptt_batches):
					start_ptr = j*global_context_size*bptt_steps
					end_ptr = (j+1)*global_context_size*bptt_steps + global_context_size - 1
					bptt_batch_x = current_clip[:, start_ptr:end_ptr, :]
					bptt_batch_y = current_clip[:, start_ptr+global_context_size:end_ptr+1, :]
					bptt_batch_m = current_mask[:, start_ptr+global_context_size:end_ptr+1, :]
					bptt_batch_loss, np_state, op, out = \
						sess.run([t_model.loss, t_model.final_state, optimizer, t_model.outputs],
							feed_dict={
								input: bptt_batch_x,
								tf_outputs: bptt_batch_y,
								tf_masks: bptt_batch_m,
								t_model.initial_state[0]:np_state[0],
								t_model.initial_state[1]:np_state[1],
								t_model.generation_phase:False
							})
					print 'epoch:', ep,'song #', i, ', bptt index:', j, ', loss:', bptt_batch_loss
					#print 'memory activations: ', 100. - ((np_state[0]<0).sum() + (np_state[1]<0).sum())/128., '%'
					#print 'out bincount:', np.bincount(np.array(out).flatten())
					#print p1 # remove
					#print p2 # remove

			# check loss on validation data
			# validation data is randomly
			# sampled from validation set

			print
			val_losses = []
			j = random.choice(range(len(vinputs)))
			current_clip = vinputs[j]
			current_mask = vmasks[j]
			n_bptt_batches = current_clip.shape[1] / (global_context_size * bptt_steps) - 1
			np_state = z_state
			for k in range(n_bptt_batches):
				start_ptr = j*global_context_size*bptt_steps
				end_ptr = (j+1)*global_context_size*bptt_steps + global_context_size - 1
				bptt_batch_x = current_clip[:, start_ptr:end_ptr, :]
				bptt_batch_y = current_clip[:, start_ptr+global_context_size:end_ptr+1, :]
				bptt_batch_m = current_mask[:, start_ptr+global_context_size:end_ptr+1, :]
				bptt_batch_loss, np_state, out = \
					sess.run([t_model.loss, t_model.final_state, t_model.outputs],
						feed_dict={
							input:bptt_batch_x,
							tf_outputs:bptt_batch_y,
							tf_masks:bptt_batch_m,
							t_model.initial_state[0]:np_state[0],
							t_model.initial_state[1]:np_state[1],
							t_model.generation_phase:False
						})
				val_losses.append(bptt_batch_loss)
				print '\033[Fminibatch ({}/{}), validation loss : {:.4f}'.format(k+1, n_bptt_batches, bptt_batch_loss)
			print 'mean validation loss:', np.mean(val_losses)

			# generate some audio


			save_path = saver.save(sess, "./params/last_model.ckpt")
			print("Model saved in file: %s\n" % save_path)
