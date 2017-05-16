import os
import numpy as np
import model
import tensorflow as tf
import random
import json

# load training and validation data
# data is randomly shuffled
files = os.listdir('../tmp/')
files = [(('../tmp/'+files[i+1]), ('../tmp/'+files[i]))  for i in range(0, len(files), 2)]
random.seed(0)
random.shuffle(files)
inputs = [np.load(files[i][0]) for i in range(len(files)-5)]
masks = [np.load(files[i][1]) for i in range(len(files)-5)]
vinputs = [np.load(files[i][0]) for i in range(len(files)-5, len(files))]
vmasks = [np.load(files[i][1]) for i in range(len(files)-5, len(files))]

# network configurations
batch_size = 64
global_context_size = 100
bptt_steps = 2
n_epochs = 200
clip_iter = 1
best_val_loss = np.inf
generation_freq = 10
iter_ = 0
start_ep = 0
start_clip = 0

# tensors to be fed to the model
input = tf.placeholder(tf.float32, [batch_size, global_context_size*bptt_steps+global_context_size-1, 1])
tf_masks = tf.placeholder(tf.float32, [batch_size, global_context_size*bptt_steps, 1])
tf_inputs = (input - 7.5)/3.75
tf_outputs = tf.placeholder(tf.uint8, [batch_size, global_context_size*bptt_steps, 1])
tf_labels = tf_masks*tf.reshape(tf.one_hot(tf_outputs, depth=16), [batch_size, global_context_size*bptt_steps, 16])
t_model = model.sample_rnn(tf_inputs, tf_labels, tf_masks, batch_size=batch_size, is_training=True)

# gradient clipping
# to prevent gradient explosion
optimizer = tf.train.AdamOptimizer(0.01)
global_step = tf.Variable(0)
gradients, v = zip(*optimizer.compute_gradients(t_model.loss))
gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
saver = tf.train.Saver()
if not os.path.exists('../params'):	os.makedirs('../params')
if not os.path.exists('../gen'):	os.makedirs('../gen')
if not os.path.exists('../logs'):	os.makedirs('../logs')


# entry point function for
# generator thread, runs gen-script.py
# and saves generated output in ./gen/*.wav
# * = <train/test>_<iter>
def generator(out_file, in_file, mask_file):
	os.system('python gen-script.py '+out_file+' '+in_file+' '+mask_file)


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
	global best_val_loss, iter_, start_ep, start_clip, inputs
	file_name = '../logs/state.log'
	file = open(file_name, 'r')
	json_ = file.read()
	file.close()
	dict_ = json.loads(json_)
	best_val_loss = dict_['loss']
	iter_ = dict_['iter']
	start_ep = dict_['epoch']
	last_clip = dict_['clip']
	if (last_clip < (len(inputs)-1)):
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
	z_state = (t_model.initial_state[0].eval(), t_model.initial_state[1].eval())
	if os.path.exists('../logs/state.log'):
		load_state()
		print 'network state restored from last saved instance ..'

	# training begins here
	for ep in range(start_ep, n_epochs):
		for i in range(start_clip, len(inputs)):
			print 'epoch #', ep
			for ci in range(clip_iter):
				print 'Training on clip #', i+1, '/', len(inputs)
				current_clip = inputs[i]
				current_mask = masks[i]
				n_bptt_batches = current_clip.shape[1] / (global_context_size * bptt_steps) - 1
				np_state = z_state
				for j in range(10):#n_bptt_batches):
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
					iter_+=1
					print 'iter:', iter_, ', bptt index:', j, ', loss:', bptt_batch_loss

			# check loss on validation data
			# validation data is randomly
			# sampled from validation set
			# saves params only is loss improves
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
			cur_val_loss = np.mean(val_losses)
			print 'mean validation loss:', cur_val_loss
			if cur_val_loss<best_val_loss:
				print 'validation loss improved! {:.4f}->{:.4f}'.format(best_val_loss, cur_val_loss)
				best_val_loss = cur_val_loss
				save_path = saver.save(sess, "../params/last_model.ckpt")
				print("Model saved in file: %s" % save_path)
			else:
				print 'validation loss did not improve.'
			dump_state(float(best_val_loss), iter_, ep, i)
			print 'state dumped at ../logs/state.log ..\n'


			# generate some audio after Training
			# on every 10 audio clips, 1 ep = 175 clips
			# approximately 17 outputs per epoch
			# 170 outputs in total for each seed
			if i%generation_freq==0:
				print '='*80
				print 'Generating sample audio ..'
				generator('train_'+str(i/generation_freq)+'.wav', files[0][0], files[0][1])
				print '='*80
