import tensorflow as tf

''' 
	architecture is 2 tier
	bptt_steps: # steps for bptt / bptt truncation threshold; default = 100 (1 second)
	global_context_size: param for global context, default = 100
	local_context_size: param for local context, default = 10
	lstm_dim: dimension of the lstm layer for global context; default = 500
	sampl_dim: dimension of down sampling layer; default = 90
	hid_dim1: dimension of 1st hidden layer; default = 100
	hid_dim2: dimension of 2nd hidden layer; default = 100
	out_dim : dimension of softmax layer; default = 16
	inputs is of shape: batch_size x (bptt_steps x global_context_size) x 1
	labels is of shape: batch_size x (bptt_steps x global_context_size) x 16
'''

class sample_rnn():
    def __init__(self, inputs, labels, bptt_steps=100, global_context_size=100, local_context_size=10, lstm_dim=500, sampl_dim=90, 
    	hid_dim1=100, hid_dim2=100, out_dim=16, batch_size=1, is_training=True):

		self.weights = {
				'sampl': tf.Variable(tf.random_uniform([lstm_dim, sampl_dim]), name='sampl/W'),
				'hidn1': tf.Variable(tf.random_uniform([sampl_dim+local_context_size, hid_dim1]), name='hidn1/W'),
				'hidn2': tf.Variable(tf.random_uniform([hid_dim1, hid_dim2]), name='hidn2/W'),
				'out': tf.Variable(tf.random_uniform([hid_dim2, out_dim]), name='out/W')
				}
		self.biases = {
				'sampl': tf.Variable(tf.random_uniform([sampl_dim]), name='sampl/b'),
				'hidn1': tf.Variable(tf.random_uniform([hid_dim1]), name='hidn1/b'),
				'hidn2': tf.Variable(tf.random_uniform([hid_dim2]), name='hidn2/b'),
				'out':	tf.Variable(tf.random_uniform([out_dim]), name='out/b')
				}

		lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_dim)
		self.initial_state = self.state = lstm_cell.zero_state(batch_size, tf.float32)
		self.total_loss = 0.
		self.mean_acc = 0.
		self.outputs = []
		self.losses = 0.
		
		with tf.variable_scope('RNN') as scope:
			for i in range(bptt_steps):
				if i>0:	scope.reuse_variables()

				global_context = inputs[:, i*global_context_size:(i+1)*global_context_size, :]
				global_context = tf.reshape(global_context, [batch_size, global_context_size])
				lstm_output, self.state = lstm_cell(global_context, self.state)
				down_sampl = tf.nn.tanh(tf.matmul(lstm_output, self.weights['sampl']) + self.biases['sampl'])
				
				print i
				for j in range(global_context_size):
					pred_index = (i+1)*global_context_size + j
					local_context =  inputs[:, pred_index-local_context_size:pred_index, :]
					local_context = tf.reshape(local_context, [batch_size, local_context_size])
					conc = tf.concat([down_sampl, local_context], axis=1)
					hid1 = tf.nn.tanh(tf.matmul(conc, self.weights['hidn1']) + self.biases['hidn1'])
					hid2 = tf.nn.tanh(tf.matmul(hid1, self.weights['hidn2']) + self.biases['hidn2'])
					out = tf.matmul(hid2, self.weights['out']) + self.biases['out']
					loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels[:, pred_index, :], logits=out))
					if j==0:	self.o = loss
					# review code from here
					correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(labels[:, pred_index, :], 1))
					accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
					self.losses+= loss
					self.mean_acc += accuracy
					if is_training is False:	self.outputs.append(tf.argmax(out, 1)[0])

			self.final_state = self.state
			self.mean_acc /= (bptt_steps*global_context_size)
			self.losses /= (bptt_steps*global_context_size)
