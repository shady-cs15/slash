import tensorflow as tf

'''
	architecture is 2 tier
	bptt_steps: # steps for bptt / bptt truncation threshold; default = 10 (1/10th second)
	global_context_size: param for global context, default = 100
	local_context_size: param for local context, default = 10
	lstm_dim: dimension of the lstm layer for global context; default = 500
	sampl_dim: dimension of down sampling layer; default = 10
	hid_dim1: dimension of 1st hidden layer; default = 80
	hid_dim2: dimension of 2nd hidden layer; default = 200
	out_dim : dimension of softmax layer; default = 16
	inputs is of shape: batch_size x (bptt_steps x global_context_size) x 1
	labels is of shape: batch_size x (bptt_steps x global_context_size) x 16
'''

class sample_rnn():
    def __init__(self, inputs, labels, masks, bptt_steps=2, global_context_size=100, local_context_size=10, lstm_dim=100, sampl_dim=10,
    	hid_dim1=80, hid_dim2=200, hid_dim3=500, out_dim=16, batch_size=1, is_training=True):

		self.weights = {
				'sampl': tf.Variable(tf.random_uniform([lstm_dim, sampl_dim]), name='sampl/W'),
				'hidn1': tf.Variable(tf.random_uniform([sampl_dim+local_context_size, hid_dim1]), name='hidn1/W'),
				'hidn2': tf.Variable(tf.random_uniform([hid_dim1, hid_dim2]), name='hidn2/W'),
				'hidn3': tf.Variable(tf.random_uniform([hid_dim2, hid_dim3]), name='hidn3/W'),
				'out': tf.Variable(tf.random_uniform([hid_dim3, out_dim]), name='out/W')
				}
		self.biases = {
				'sampl': tf.Variable(tf.random_uniform([sampl_dim]), name='sampl/b'),
				'hidn1': tf.Variable(tf.random_uniform([hid_dim1]), name='hidn1/b'),
				'hidn2': tf.Variable(tf.random_uniform([hid_dim2]), name='hidn2/b'),
				'hidn3': tf.Variable(tf.random_uniform([hid_dim3]), name='hidn3/b'),
				'out':	tf.Variable(tf.random_uniform([out_dim]), name='out/b')
				}

		lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_dim)
		self.initial_state = self.state = lstm_cell.zero_state(batch_size, tf.float32)
		self.total_loss = 0.
		self.mean_acc = 0.
		self.outputs = []
		self.loss = 0.
		count = 0

		with tf.variable_scope('RNN') as scope:
			for i in range(bptt_steps):
				if i>0:	scope.reuse_variables()

				global_context = inputs[:, i*global_context_size:(i+1)*global_context_size, :]
				global_context = tf.reshape(global_context, [batch_size, global_context_size])
				lstm_output, self.state = lstm_cell(global_context, self.state)
				lstm_output = tf.nn.relu(lstm_output)
				down_sampl = (tf.matmul(lstm_output, self.weights['sampl']) + self.biases['sampl'])
				down_sampl = down_sampl/10.

				#if i==0:	self.o1, self.o2 = lstm_output, down_sampl # remove
				print i
				for j in range(global_context_size):
					pred_index = (i+1)*global_context_size + j
					local_context =  inputs[:, pred_index-local_context_size:pred_index, :]
					local_context = tf.reshape(local_context, [batch_size, local_context_size])
					conc = tf.concat([down_sampl, local_context], axis=1)
					hid1 = tf.nn.relu(tf.matmul(conc, self.weights['hidn1']) + self.biases['hidn1'])
					hid2 = tf.nn.relu(tf.matmul(hid1, self.weights['hidn2']) + self.biases['hidn2'])
					hid3 = tf.nn.relu(tf.matmul(hid2, self.weights['hidn3']) + self.biases['hidn3'])
					out = tf.matmul(hid3, self.weights['out']) + self.biases['out']
					out = tf.multiply(out, masks[:, pred_index - global_context_size])
					loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels[:, pred_index-global_context_size, :], logits=out))
					correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(labels[:, pred_index - global_context_size, :], 1))
					accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32)) - (batch_size - tf.reduce_sum(masks[:, pred_index-global_context_size, 0]))
					mean_acc = accuracy/tf.reduce_sum(masks[:, pred_index-global_context_size, 0])
					self.loss += loss
					count+= tf.reduce_mean(masks[:, pred_index - global_context_size, 0])
					self.mean_acc += mean_acc
					if is_training is False:	self.outputs.append(tf.argmax(out, 1))

			self.final_state = self.state
			self.mean_acc /= (bptt_steps*global_context_size)
			self.loss /= count
