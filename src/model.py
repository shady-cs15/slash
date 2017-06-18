import tensorflow as tf

class sample_rnn():
	def __init__(self, inputs, labels, bptt_steps=64, context_size=16, dim=(256, 512), q_levels=16, batch_size=64, n_rnn=3, n_mlp=3, generator=False):

		self.bptt_steps = bptt_steps
		self.context_size = context_size
		self.dim = dim
		self.q_levels = q_levels
		self.batch_size = batch_size
		self.n_rnn = n_rnn
		self.n_mlp = n_mlp

		# function to bound the initialisation weights
		def w_b(n_in, n_out):
			return (2./(n_in+n_out))**0.5

		# dictionary of weights and biases for fully connected layers
		with tf.variable_scope('sample_rnn') as scope:
			if generator==True:
				tf.get_variable_scope().reuse_variables()
			self.W = {
				'samp': tf.get_variable('samp/W', [dim[0], dim[0]], tf.float32, tf.random_uniform_initializer(-w_b(dim[0], dim[0]), w_b(dim[0], dim[0]))),
				'mlp0': tf.get_variable('mlp0/W', [2*context_size, dim[1]], tf.float32, tf.random_uniform_initializer(-w_b(2*context_size, dim[1]), w_b(2*context_size, dim[1]))),
				'mlp1': tf.get_variable('mlp1/W', [dim[1], dim[1]], tf.float32, tf.random_uniform_initializer(-w_b(dim[1], dim[1]), w_b(dim[1], dim[1]))),
				'mlp2': tf.get_variable('mlp2/W', [dim[1], q_levels], tf.float32, tf.random_uniform_initializer(-w_b(dim[1], q_levels), w_b(dim[1], q_levels)))
			}
			self.b = {
				'samp': tf.get_variable('samp/b', [dim[0]], tf.float32, tf.constant_initializer(0.0)),
				'mlp0': tf.get_variable('mlp0/b', [dim[1]], tf.float32, tf.constant_initializer(0.0)),
				'mlp1': tf.get_variable('mlp1/b', [dim[1]], tf.float32, tf.constant_initializer(0.0)),
				'mlp2': tf.get_variable('mlp2/b', [q_levels], tf.float32, tf.constant_initializer(0.0))
			}

		def stacked_mlps(inputs, n_mlps):
			assert n_mlps<=3
			mlp0_out = tf.nn.relu(tf.matmul(inputs, self.W['mlp0']) + self.b['mlp0'])
			if n_mlps==1:	return mlp0_out
			mlp1_out = tf.nn.relu(tf.matmul(mlp0_out, self.W['mlp1']) + self.b['mlp1'])
			if n_mlps==2:	return mlp1_out
			return tf.nn.relu(tf.matmul(mlp1_out, self.W['mlp2']) + self.b['mlp2'])


		cell = tf.contrib.rnn.BasicLSTMCell(dim[0])
		stacked_lstm = tf.contrib.rnn.MultiRNNCell([cell] * n_rnn)
		self.initial_state = self.state = stacked_lstm.zero_state(batch_size, tf.float32)
		self.loss = []

		print 'Building computation graph..\n'
		with tf.variable_scope('sample_rnn') as scope:
			for i in range(bptt_steps):
				if generator==True:	scope.reuse_variables()
				elif i>0:	scope.reuse_variables()

				lstm_inp = inputs[:, i*context_size:(i+1)*context_size, :]
				lstm_inp = tf.reshape(lstm_inp, [batch_size, context_size])
				lstm_out, self.state = stacked_lstm(lstm_inp, self.state)
				emb = tf.matmul(lstm_out, self.W['samp']) + self.b['samp']

				print '\033[FGraph built: {:.2f} %'.format(i*100./bptt_steps)
				for j in range(context_size):
					global_context = emb[:, j*context_size:(j+1)*context_size]
					pred_index = (i+1)*context_size + j
					local_context =  inputs[:, pred_index-context_size:pred_index, :]
					local_context = tf.reshape(local_context, [batch_size, context_size])
					context = tf.concat([global_context, local_context], 1)
					mlps_out = stacked_mlps(context, n_mlp)

					# loss
					label = labels[:, pred_index-context_size, :]
					loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=mlps_out))
					self.loss.append(loss)

		print 'computation graph built..'
		self.final_state = self.state
		self.loss = tf.reduce_mean(self.loss)
