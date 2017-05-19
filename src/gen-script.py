import os
import numpy as np
import model
import tensorflow as tf
import data_utils as du
import random
import sys

out_file = sys.argv[1]
gen_indx = int(sys.argv[2])
data = np.load('../tmp/data.npy')[gen_indx].reshape(1, 2, 80000)
inp_x = data[:, 0, :].reshape(1, int(8e4), 1)
inp_m = data[:, 1, :].reshape(1, int(8e4), 1)

global_context_size = 100
batch_size = 1
input = tf.placeholder(tf.float32, [batch_size, (global_context_size*2)-1, 1])
tf_inputs = (input - 7.5)/7.5
tf_outputs = tf.placeholder(tf.uint8, [batch_size, global_context_size, 1])
tf_masks = tf.placeholder(tf.float32, [batch_size, global_context_size, 1])
tf_labels = tf_masks*tf.reshape(tf.one_hot(tf_outputs, depth=16), [batch_size, global_context_size, 16])
g_model = model.sample_rnn(tf_inputs, tf_labels, tf_masks, bptt_steps=1, batch_size=1, is_training=False)
if not os.path.exists('../gen'):	os.makedirs('../gen')
saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, '../params/best_model.ckpt')
	print 'model restored..'
	np_state = (g_model.initial_state[0].eval(), g_model.initial_state[1].eval())
	predictions = []

	# prediction phase
	print 'Warming up rnn ..\n'
	n_pred_batches = 50000/global_context_size - 1
	for i in range(n_pred_batches):
		cur_input = inp_x[:, i*global_context_size:(i+2)*global_context_size-1, :]
		cur_label = inp_x[:, (i+1)*global_context_size:(i+2)*global_context_size, :]
		cur_mask = inp_m[:, (i+1)*global_context_size:(i+2)*global_context_size, :]
		loss, np_state, out = sess.run([g_model.loss, g_model.final_state, g_model.outputs],
			feed_dict = {
				input:cur_input,
				tf_outputs:cur_label,
				tf_masks:cur_mask,
				g_model.initial_state[0]:np_state[0],
				g_model.initial_state[1]:np_state[1],
				g_model.generation_phase:False
			})
		print '\033[Findex: ({}/{}), loss: {:.4f}'.format(i+1, n_pred_batches, loss)
		predictions += list(np.array(out).flatten())


	# generation phase
	gen_len = 5
	print 'Generating the next', gen_len, 'seconds ..'
	for i in range(100*gen_len):
		cur_input = np.array(np.concatenate([predictions[-global_context_size:], np.zeros([global_context_size-1])])).reshape([1, 2*global_context_size-1, 1])
		np_state, out = sess.run([g_model.final_state, g_model.outputs],
			feed_dict = {
				input:cur_input,
				tf_outputs:cur_label,
				tf_masks:cur_mask,
				g_model.initial_state[0]:np_state[0],
				g_model.initial_state[1]:np_state[1],
				g_model.generation_phase:True
			})
		predictions += list(np.array(out, dtype=np.uint8).flatten())

	du.save_file('../gen/'+out_file, np.array(predictions))
	print 'generated file saved in ../gen/'+out_file+'!'
