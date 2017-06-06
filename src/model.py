import tensorflow as tf

#	architecture is 2 tier
#	bptt_steps: # steps for bptt / bptt truncation threshold; default = 10 (1/10th second)
#	global_context_size: param for global context, default = 100
#	local_context_size: param for local context, default = 10
#	lstm_dim: dimension of the lstm layer for global context; default = 500
#	sampl_dim: dimension of down sampling layer; default = 10
#	hid_dim1: dimension of 1st hidden layer; default = 80
#	hid_dim2: dimension of 2nd hidden layer; default = 200
#	hid_dim3: dimension of 3rd hidden layer; default = 500
#	out_dim : dimension of softmax layer; default = 16
#	inputs is of shape: batch_size x (bptt_steps x global_context_size) x 1
#	labels is of shape: batch_size x (bptt_steps x global_context_size) x 16


class sample_rnn():
	def __init__(self, inputs, labels, masks, bptt_steps=2, global_context_size=100, local_context_size=10, lstm_dim=100,
	sampl_dim=10, hid_dim1=80, hid_dim2=200, hid_dim3=500, out_dim=16, batch_size=1, is_training=True):

		def lu(f):
			return tf.nn.dropout(f, 1.)

		def w_b(n_in, n_out):
			return (2./(n_in+n_out))**0.5

		self.weights = {
				'sampl': tf.Variable(tf.random_uniform([lstm_dim, sampl_dim], -w_b(lstm_dim, sampl_dim), w_b(lstm_dim, sampl_dim)), name='sampl/W'),
				'hidn1': tf.Variable(tf.random_uniform([sampl_dim+local_context_size, hid_dim1], -w_b(sampl_dim+local_context_size, hid_dim1), w_b(sampl_dim+local_context_size, hid_dim1)), name='hidn1/W'),
				'hidn2': tf.Variable(tf.random_uniform([hid_dim1, hid_dim2], -w_b(hid_dim1, hid_dim2), w_b(hid_dim1, hid_dim2)), name='hidn2/W'),
				'hidn3': tf.Variable(tf.random_uniform([hid_dim2, hid_dim3], -w_b(hid_dim2, hid_dim3), w_b(hid_dim2, hid_dim3)), name='hidn3/W'),
				'out': tf.Variable(tf.random_uniform([hid_dim3, out_dim], -w_b(hid_dim3, out_dim), w_b(hid_dim3, out_dim)), name='out/W')
				}

		self.biases = {
				'sampl': tf.Variable(tf.zeros([sampl_dim]), name='sampl/b'),
				'hidn1': tf.Variable(tf.zeros([hid_dim1]), name='hidn1/b'),
				'hidn2': tf.Variable(tf.zeros([hid_dim2]), name='hidn2/b'),
				'hidn3': tf.Variable(tf.zeros([hid_dim3]), name='hidn3/b'),
				'out':	tf.Variable(tf.zeros([out_dim]), name='out/b')
				}

		lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_dim)
		self.initial_state = self.state = lstm_cell.zero_state(batch_size, tf.float32)
		self.total_loss = 0.
		self.mean_acc = 0.
		self.outputs = []
		self.loss = 0.
		count = 0
		self.generation_phase = tf.placeholder(tf.bool)
		inputs = tf.cond(self.generation_phase, lambda:inputs[:,:global_context_size,:], lambda:inputs)

		keep_prob = 0.3
		global_mask = tf.concat([tf.ones(keep_prob*global_context_size), tf.zeros((1-keep_prob)*global_context_size)], 0)
		local_mask = tf.concat([tf.ones(keep_prob*local_context_size), tf.zeros((1-keep_prob)*local_context_size)], 0)

		print 'Building computation graph ..'
		with tf.variable_scope('RNN') as scope:
			for i in range(bptt_steps):
				if i>0:	scope.reuse_variables()

				global_context = inputs[:, i*global_context_size:(i+1)*global_context_size, :]
				global_context = tf.reshape(global_context, [batch_size, global_context_size])
				lstm_output, self.state = lstm_cell(global_context, self.state)
				down_sampl = tf.nn.tanh(tf.matmul(lstm_output, self.weights['sampl']) + self.biases['sampl'])

				if i>0:
					last_out = (tf.cast(tf.concat(self.outputs[-global_context_size:], 1), tf.float32) - 7.5)/7.5
					global_mask = tf.random_shuffle(global_mask)
					global_context = global_context*global_mask + last_out*(1-global_mask)

				print 'Graph built:', i*100./bptt_steps, '%'
				for j in range(global_context_size):
					pred_index = (i+1)*global_context_size + j
					local_context =  inputs[:, pred_index-local_context_size:pred_index, :]
					local_context = tf.reshape(local_context, [batch_size, local_context_size])

					if i>0:
						last_local_out = (tf.cast(tf.concat(self.outputs[-local_context_size:], 1), tf.float32) - 7.5)/7.5
						local_mask = tf.random_shuffle(local_mask)
						local_context = local_context*local_mask + last_local_out*(1-local_mask)

					conc = tf.concat([down_sampl, local_context], axis=1)
					hid1 = tf.nn.relu(tf.matmul(conc, self.weights['hidn1']) + self.biases['hidn1'])
					hid2 = tf.nn.relu(tf.matmul(hid1, self.weights['hidn2']) + self.biases['hidn2'])
					hid3 = tf.nn.relu(tf.matmul(hid2, self.weights['hidn3']) + self.biases['hidn3'])
					out = tf.matmul(hid3, self.weights['out']) + self.biases['out']
					out = tf.multiply(out, masks[:, pred_index - global_context_size])

					# loss
					label = labels[:, pred_index-global_context_size, :]
					loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels[:, pred_index-global_context_size, :], logits=out))
					self.loss += loss
					count += tf.reduce_mean(masks[:, pred_index - global_context_size, 0])

					# sampling
					sample = tf.multinomial(tf.log(tf.nn.softmax(out)), 1)
					self.outputs.append(tf.reshape(sample, [batch_size, 1]))

					if is_training is False:
						last_pred = (tf.cast(sample, tf.float32) -7.5)/7.5
						last_pred = tf.reshape(last_pred, [1, 1, 1])
						inputs = tf.cond(self.generation_phase, lambda: tf.concat([inputs, last_pred], axis=1), lambda: inputs)

			print 'Graph built:', 100.0, '%'
			self.final_state = self.state
			self.loss /= count
