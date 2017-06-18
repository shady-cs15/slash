import os
import numpy as np
import model
import tensorflow as tf
import random
import json
from ConfigParser import ConfigParser as Config
from operator import mul
import data_utils as du

# read configurations
cfg = Config()
cfg.read('settings.cfg')
batch_size = int(cfg.get('model', 'batch_size'))
bptt_steps = int(cfg.get('model', 'bptt_steps'))
context_size = int(cfg.get('model', 'context_size'))
n_epochs = int(cfg.get('train', 'n_epochs'))
gen_freq = int(cfg.get('train', 'gen_freq'))
val_freq = int(cfg.get('train', 'val_freq'))
q_levels = int(cfg.get('process', 'q_levels'))
seq_len = int(cfg.get('process', 'seq_len'))
n_samples = int(cfg.get('train', 'n_samples'))
bitrate = int(cfg.get('process', 'bitrate'))
train_summary_freq = int(cfg.get('train', 'train_summary_freq'))

# load training and validation data
# data is randomly shuffled
# reshape into n_batches x batch_size x nsteps x 1
print '='*80
inputs = np.load('../tmp/data.npy')
np.random.seed(23455)
np.random.shuffle(inputs)
n_train_batches = (inputs.shape[0] - batch_size)/batch_size
train_inputs = inputs[:n_train_batches*batch_size].reshape(n_train_batches, batch_size, seq_len*int(1e4), 1)
valid_inputs = inputs[-batch_size:].reshape(1, batch_size, seq_len*int(1e4), 1)
print 'train data shape:', train_inputs.shape
print 'valid data shape:', valid_inputs.shape


best_val_loss = np.inf
iter_ = 0
start_ep = 0
start_batch = 0
mean = (q_levels-1)/2.


# tensors to be fed to the model
input = tf.placeholder(tf.float32, [batch_size, context_size*bptt_steps+context_size-1, 1])
tf_inputs = (input- mean)/mean
tf_outputs = tf.placeholder(tf.uint8, [batch_size, context_size*bptt_steps, 1])
tf_labels = tf.reshape(tf.one_hot(tf_outputs, depth=q_levels), [batch_size, context_size*bptt_steps, q_levels])
t_model = model.sample_rnn(tf_inputs, tf_labels, batch_size=batch_size, bptt_steps=bptt_steps)

# gradient clipping
# to prevent gradient explosion
optimizer = tf.train.AdamOptimizer(0.001)
global_step = tf.Variable(0, trainable=False)
gradients, v = zip(*optimizer.compute_gradients(t_model.loss))
gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
saver = tf.train.Saver()
if not os.path.exists('../params'):	os.makedirs('../params')
if not os.path.exists('../gen'):	os.makedirs('../gen')
if not os.path.exists('../logs'):	os.makedirs('../logs')
if not os.path.exists('../logs/train'):	os.makedirs('../logs/train')
if not os.path.exists('../logs/valid'):	os.makedirs('../logs/valid')

#summary
tf.summary.scalar('loss', t_model.loss)
summary_op = tf.summary.merge_all()

print '='*80
print 'architecture of train model..'
total_params = 0
tvars = tf.trainable_variables()
for var in tvars:
	print var.name, var.get_shape()
	total_params += int(np.prod(list(var.get_shape())))
print 'total params to be learnt: ', total_params
print '='*80

# saves generated output in ./gen/*.wav
# iter_id.wav
print 'Generator model..'
g_input = tf.placeholder(tf.float32, [n_samples, context_size*2-1, 1])
gtf_inputs = (g_input- mean)/mean
gtf_outputs = tf.placeholder(tf.uint8, [n_samples, context_size, 1])
gtf_labels = tf.reshape(tf.one_hot(gtf_outputs, depth=q_levels), [n_samples, context_size, q_levels])
g_model = model.sample_rnn(gtf_inputs, gtf_labels, batch_size=n_samples, bptt_steps=1, generator=True)
print '='*80

def generate_samples(iter_, sess, gen_path='../gen/'):
	global inputs, context_size, n_samples, g_model, gtf_inputs, gtf_outputs, gtf_labels
	indices = random.sample(range(inputs.shape[0]), n_samples)
	data = []
	seed_length = int(1e3)
	for i in range(n_samples):
		cur_seed = inputs[indices[i], :seed_length].reshape(1, seed_length, 1)
		data.append(cur_seed)
	data = np.concatenate(data, 0)

	n_pred_batches = seed_length/context_size - 1
	g_np_state = sess.run(g_model.initial_state)

	print 'warming up rnn..'
	for j in range(n_pred_batches):
		cur_batch = data[:, j*context_size:(j+2)*context_size-1, :]
		cur_label = data[:, (j+1)*context_size:(j+2)*context_size, :]
		loss, g_np_state = sess.run([g_model.loss, g_model.final_state],
			feed_dict = {
				g_input:cur_batch,
				gtf_outputs:cur_label,
				g_model.initial_state:g_np_state,
				g_model.generation_phase:False
			})

	print 'generating sequences..'
	outputs = []
	n_gen_batches = int(5e4)/context_size
	input = cur_batch[:, -context_size:, :]
	label = cur_label
	for i in range(n_gen_batches):
		input = np.concatenate([input, np.zeros([n_samples, context_size-1, 1], np.float32)], axis=1)
		g_np_state, output = sess.run([g_model.final_state, g_model.outputs],
			feed_dict = {
				g_input:input,
				gtf_outputs:label,
				g_model.initial_state:g_np_state,
				g_model.generation_phase:True
			}
		)
		#print input[0].flatten()
		#print output[0].flatten()
		input = output[:, -context_size:, :].astype(np.uint8)
		outputs.append(input)
	outputs = np.concatenate(outputs, axis=1)

	for i in range(outputs.shape[0]):
		out_file = str(iter_)+'_'+str(i)
		du.save_file(gen_path+out_file, outputs[i].flatten(), bitrate, q_levels)

# functions to dump and load state of the network
# state includes (before saving weights) -
# 1. best_val_loss, 2. iter_, 3. ep, 4. batch
# states are dumped into ../logs/state.log
def dump_state(loss, iter, epoch, batch):
	file_name = '../logs/state.log'
	dict_ = {
		'loss': loss,
		'iter': iter,
		'epoch': epoch,
		'batch': batch
	}
	json_ = json.dumps(dict_)
	file = open(file_name, 'wb')
	file.write(json_)
	file.close()

def load_state():
	global best_val_loss, iter_, start_ep, start_batch, train_inputs
	file_name = '../logs/state.log'
	file = open(file_name, 'r')
	json_ = file.read()
	file.close()
	dict_ = json.loads(json_)
	best_val_loss = dict_['loss']
	iter_ = dict_['iter']
	start_ep = dict_['epoch']
	last_batch = dict_['batch']
	if (last_batch < (train_inputs.shape[0]-1)):
		start_batch = last_batch+1
	else:
		start_ep +=1
		start_batch = 0


# tensorflow Session
# begins here
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	train_writer = tf.summary.FileWriter('../logs/train', tf.get_default_graph())
	valid_writer = tf.summary.FileWriter('../logs/valid')

	# load state of training
	# and parameters of the neural network
	if os.path.exists('../params/last_model.ckpt.meta'):
		saver.restore(sess, '../params/last_model.ckpt')
		print 'model restored from last checkpoint ..'
	elif os.path.exists('../params/best_model.ckpt.meta'):
		saver.restore(sess, '../params/best_model.ckpt')
		print 'model restored from last checkpoint ..'

	z_state = sess.run(t_model.initial_state)
	if os.path.exists('../logs/state.log'):
		load_state()
		print 'network state restored from last saved instance ..'

	# training begins here
	train_losses = []
	for ep in range(start_ep, n_epochs):
		for i in range(start_batch, train_inputs.shape[0]):
			print '='*80
			print 'epoch #', ep+1
			print 'Training on batch #', i+1, '/', train_inputs.shape[0]
			current_batch = train_inputs[i]
			n_bptt_batches = current_batch.shape[1] / (context_size * bptt_steps) - 1
			np_state = z_state

			for j in range(n_bptt_batches):
				start_ptr = j*context_size*bptt_steps
				end_ptr = (j+1)*context_size*bptt_steps + context_size - 1
				bptt_batch_x = current_batch[:, start_ptr:end_ptr, :]
				bptt_batch_y = current_batch[:, start_ptr+context_size:end_ptr+1, :]
				bptt_batch_loss, np_state, op, summary = \
						sess.run([t_model.loss, t_model.final_state, optimizer, summary_op],
							feed_dict={
								input: bptt_batch_x,
								tf_outputs: bptt_batch_y,
								t_model.initial_state:np_state,
								t_model.generation_phase:False
							})
				iter_+=1
				train_losses.append(bptt_batch_loss)

				# check and write summaries
				if (iter_+1)%train_summary_freq==0:
					print 'iter:', iter_, ', loss:', bptt_batch_loss
					train_writer.add_summary(summary, iter_+1)

				# check loss on validation data
				if (iter_+1)%val_freq==0:
					print '='*80
					print
					val_losses = []
					for j in range(valid_inputs.shape[0]):
						current_batch = valid_inputs[j]
						n_bptt_batches = current_batch.shape[1] / (context_size * bptt_steps) - 1
						np_state = z_state
						for k in range(n_bptt_batches):
							start_ptr = j*context_size*bptt_steps
							end_ptr = (j+1)*context_size*bptt_steps + context_size - 1
							bptt_batch_x = current_batch[:, start_ptr:end_ptr, :]
							bptt_batch_y = current_batch[:, start_ptr+context_size:end_ptr+1, :]
							bptt_batch_loss, np_state, summary = \
								sess.run([t_model.loss, t_model.final_state, summary_op],
									feed_dict={
										input:bptt_batch_x,
										tf_outputs:bptt_batch_y,
										t_model.initial_state:np_state,
										t_model.generation_phase:False
									})
							val_losses.append(bptt_batch_loss)
							if (j*n_bptt_batches+k+1)%200==0.0:
								print '\033[Fminibatch ({}/{}), validation loss : {:.7f}'.format((j)*n_bptt_batches+k+1, n_bptt_batches*valid_inputs.shape[0], bptt_batch_loss)
					cur_val_loss = np.mean(val_losses)
					valid_writer.add_summary(summary, iter_+1)
					mean_train_loss = np.mean(train_losses)
					train_losses = []
					print 'mean train loss:', mean_train_loss
					print 'mean validation loss:', cur_val_loss
					print 'best validation loss so far:', best_val_loss

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
					print '='*80



				# generate some audio after Training
				if (iter_+1)%gen_freq==0:
					print 'Generating sample audio ..'
					generate_samples(iter_, sess)
					print '='*80

		start_clip=0
