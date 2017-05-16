import os
import numpy as np
import model
import tensorflow as tf
import data_utils as du


files = os.listdir('../tmp/')
files = [(('../tmp/'+files[i+1]), ('../tmp/'+files[i]))  for i in range(0, len(files), 2)]
random.seed(0)
random.shuffle(files)
val_list = files[-5:]

inp_x = np.load(val_list[0][0])[0].reshape(1, 50000, 1)
inp_m = np.load(val_list[0][1])[0].reshape(1, 50000, 1)

global_context_size = 100
batch_size = 1
input = tf.placeholder(tf.float32, [batch_size, (global_context_size*2)-1, 1])
tf_inputs = (input - 7.5)/3.75
tf_outputs = tf.placeholder(tf.uint8, [batch_size, global_context_size, 1])
tf_masks = tf.placeholder(tf.float32, [batch_size, global_context_size, 1])
tf_labels = tf_masks*tf.reshape(tf.one_hot(tf_outputs, depth=16), [batch_size, global_context_size, 16])
g_model = model.sample_rnn(tf_inputs, tf_labels, tf_masks, bptt_steps=1, batch_size=1, is_training=False)

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, './params/last_model.ckpt')
	print 'model restored..'
	np_state = (g_model.initial_state[0].eval(), g_model.initial_state[1].eval())
	predictions = []

	# prediction phase
	n_pred_batches = 20000/global_context_size - 1
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
		print 'index:', i, 'loss:', loss
		predictions += list(np.array(out).flatten())


	# generation phase
	for i in range(500):
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
		#print np.array(predictions[-global_context_size:]), np.array(out).flatten()
		predictions += list(np.array(out, dtype=np.uint8).flatten())

	du.save_file('out.wav', np.array(predictions))
