import os
import numpy as np
import model
import tensorflow as tf
import random
import json

# load training and validation data
# data is randomly shuffled
data = np.load('../tmp/data.npy')
np.random.seed(23455)
np.random.shuffle(data)
inputs = data[:, :1, :].transpose(0, 2, 1)
masks = data[:, 1:, :].transpose(0, 2, 1)


# reshape into n_batches x batch_size x nsteps x 1
batch_size = 64
n_train_batches = (inputs.shape[0] - batch_size)/batch_size
train_inputs = inputs[:n_train_batches*batch_size].reshape(n_train_batches, batch_size, 8*int(1e4), 1)
train_masks = masks[:n_train_batches*batch_size].reshape(n_train_batches, batch_size, 8*int(1e4), 1)
valid_inputs = inputs[-batch_size:].reshape(1, batch_size, 8*int(1e4), 1)
valid_masks = masks[-batch_size:].reshape(1, batch_size, 8*int(1e4), 1)
print 'train data shape:', train_inputs.shape
print 'valid data shape:', valid_inputs.shape


# network configurations
global_context_size = 100
bptt_steps = 10
n_epochs = 300
clip_iter = 1
best_val_loss = np.inf
generation_freq = 14
validation_freq = 7
iter_ = 0
start_ep = 0
start_clip = 0


# tensors to be fed to the model
input = tf.placeholder(tf.float32, [batch_size, global_context_size*bptt_steps+global_context_size-1, 1])
tf_masks = tf.placeholder(tf.float32, [batch_size, global_context_size*bptt_steps, 1])
tf_inputs = (input- 7.5)/7.5
tf_outputs = tf.placeholder(tf.uint8, [batch_size, global_context_size*bptt_steps, 1])
tf_labels = tf_masks*tf.reshape(tf.one_hot(tf_outputs, depth=256), [batch_size, global_context_size*bptt_steps, 256])
t_model = model.sample_rnn(tf_inputs, tf_labels, tf_masks, batch_size=batch_size, bptt_steps=bptt_steps, is_training=True)


# gradient clipping
# to prevent gradient explosion
optimizer = tf.train.AdamOptimizer(0.01)
global_step = tf.Variable(0)
gradients, v = zip(*optimizer.compute_gradients(t_model.loss))
gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
saver = tf.train.Saver()
if not os.path.exists('../params'):	os.makedirs('../params')
if not os.path.exists('../gen'):	os.makedirs('../gen')
if not os.path.exists('../logs'):	os.makedirs('../logs')


# entry point function for
# generator thread, runs gen-script.py
# and saves generated output in ./gen/*.wav
# * = <train/test>_<iter>
def generator(out_file, gen_indx):
	os.system('python gen-script.py '+out_file+' '+str(gen_indx))


# functions to dump and load state of the network
# state includes (before saving weights) -
# 1. best_val_loss, 2. iter_, 3. ep, 4. i
# states are dumped into ../logs/state.log
def dump_state(loss, iter, epoch, clip):
	file_name = '../logs/state.log'
	dict_ = {
		'loss': loss,
		'iter': iter,
		'epoch': epoch,
		'clip': clip
	}
	json_ = json.dumps(dict_)
	file = open(file_name, 'wb')
	file.write(json_)
	file.close()

def load_state():
	global best_val_loss, iter_, start_ep, start_clip, train_inputs
	file_name = '../logs/state.log'
	file = open(file_name, 'r')
	json_ = file.read()
	file.close()
	dict_ = json.loads(json_)
	best_val_loss = dict_['loss']
	iter_ = dict_['iter']
	start_ep = dict_['epoch']
	last_clip = dict_['clip']
	if (last_clip < (train_inputs.shape[0]-1)):
		start_clip = last_clip+1
	else:
		start_ep +=1
		start_clip = 0


# tensorflow Session
# begins here
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# load state of training
	# and parameters of the neural network
	if os.path.exists('../params/last_model.ckpt.meta'):
		saver.restore(sess, '../params/last_model.ckpt')
		print 'model restored from last checkpoint ..'
	elif os.path.exists('../params/best_model.ckpt.meta'):
		saver.restore(sess, '../params/best_model.ckpt.meta')
		print 'model restored from last checkpoint ..'

	z_state = (t_model.initial_state[0].eval(), t_model.initial_state[1].eval())
	if os.path.exists('../logs/state.log'):
		load_state()
		print 'network state restored from last saved instance ..'

	# training begins here
	for ep in range(start_ep, n_epochs):
		for i in range(start_clip, train_inputs.shape[0]):
			print '\nepoch #', ep+1
			for ci in range(clip_iter):
				print 'Training on batch #', i+1, '/', train_inputs.shape[0]
				current_clip = train_inputs[i]
				current_mask = train_masks[i]
				n_bptt_batches = current_clip.shape[1] / (global_context_size * bptt_steps) - 1
				np_state = z_state
				for j in range(n_bptt_batches):
					start_ptr = j*global_context_size*bptt_steps
					end_ptr = (j+1)*global_context_size*bptt_steps + global_context_size - 1
					bptt_batch_x = current_clip[:, start_ptr:end_ptr, :]
					bptt_batch_y = current_clip[:, start_ptr+global_context_size:end_ptr+1, :]
					bptt_batch_m = current_mask[:, start_ptr+global_context_size:end_ptr+1, :]
					bptt_batch_loss, np_state, op = \
						sess.run([t_model.loss, t_model.final_state, optimizer],
							feed_dict={
								input: bptt_batch_x,
								tf_outputs: bptt_batch_y,
								tf_masks: bptt_batch_m,
								t_model.initial_state[0]:np_state[0],
								t_model.initial_state[1]:np_state[1],
								t_model.generation_phase:False
							})
					iter_+=1
					print 'iter:', iter_, ', bptt index:', j+1, ', loss:', bptt_batch_loss
					#print np.max(o2), np.min(o2), np.mean(o2)
					#print np.sum(o1>=0.5)#, o2
					#print np.sum((o1<0.5) & (o1>=0.))
					#print np.sum((o1>=-0.5) & (o1<0.))
					#print np.sum((o1<-0.5))
					#print np.bincount(np.array(out).flatten())

			# check loss on validation data
			# validation data is randomly
			# sampled from validation set
			# saves params only is loss improves
			if (i+1)%validation_freq==0:
				print
				val_losses = []
				for j in range(valid_inputs.shape[0]):
					current_clip = valid_inputs[j]
					current_mask = valid_masks[j]
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
						print '\033[Fminibatch ({}/{}), validation loss : {:.7f}'.format((j)*n_bptt_batches+k+1, n_bptt_batches*valid_inputs.shape[0], bptt_batch_loss)
				cur_val_loss = np.mean(val_losses)
				print 'mean validation loss:', cur_val_loss
				if cur_val_loss<best_val_loss:
					print 'validation loss improved! {:.4f}->{:.4f}'.format(best_val_loss, cur_val_loss)
					best_val_loss = cur_val_loss
					save_path = saver.save(sess, "../params/best_model.ckpt")
					print("Model saved in file: %s" % save_path)
				else:
					print 'validation loss did not improve.'
					save_path = saver.save(sess, "../params/last_model.ckpt")
					print("Model saved in file: %s" % save_path)
				dump_state(float(best_val_loss), iter_, ep, i)
				print 'state dumped at ../logs/state.log ..'

			# generate some audio after Training
			# on every 25 batches, 1 ep = 50 batches
			# approximately 3 outputs per epoch
			# 900 outputs in total for each seed
			#if (i+1)%generation_freq==0:
			#	print '='*80
			#	print 'Generating sample audio ..'
			#	generator('valid_'+str(i/generation_freq)+'.wav', inputs.shape[0]-batch_size)
			#	print '='*80
		start_clip=0
