import os
import numpy as np
import model
import tensorflow as tf

inp1_x = np.load('../data/gf1.npy')[:20000]
inp1_y = np.eye(16)[inp1_x]
inp1_x = ((inp1_x - 7.5)/7.5).reshape(1, 20000, 1)
inp1_y = inp1_y.reshape(1, 20000, 16)

inp2_x = np.load('../data/gf5.npy')[:20000]
inp2_y = np.eye(16)[inp2_x]
inp2_x = ((inp2_x - 7.5)/7.5).reshape(1, 20000, 1)
inp2_y = inp2_y.reshape(1, 20000, 16)

print inp1_x, inp1_y
print inp2_x, inp2_y


global_context_size = 100
input = tf.placeholder(tf.float32, [1, 2*global_context_size-1, 1])
label = tf.placeholder(tf.uint8, [1, global_context_size, 16])
g_model = model.sample_rnn(input, label, bptt_steps=1, is_training=False)
saver = tf.train.Saver()
seed_x = inp1_x
seed_y = inp1_y

with tf.Session() as sess:
	saver.restore(sess, './params/last_model.ckpt')
	print 'model restored..'
	np_state = (g_model.initial_state[0].eval(), g_model.initial_state[1].eval())

	# prediction phase
	n_pred_batches = 20000/global_context_size - 1
	for i in range(n_pred_batches):
		cur_input = seed_x[:, i*global_context_size:(i+2)*global_context_size-1, :]
		cur_label = seed_y[:, (i+1)*global_context_size:(i+2)*global_context_size, :]
		loss, acc, np_state, out = sess.run([g_model.loss, g_model.mean_acc, g_model.final_state, g_model.outputs], 
			feed_dict={input:cur_input, label:cur_label, g_model.initial_state[0]:np_state[0], g_model.initial_state[1]:np_state[1]})
		print 'index:', i, 'loss:', loss, 'accuracy:', acc
		print out
