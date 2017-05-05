import os
import numpy as np
from etaprogress.progress import ProgressBar as pgb

#===============================================================================
# Pre processing script for batch generation
# Reads quantized raw waves from ../data/*.npy files
# Generates batches and masks
# Stores them in ../tmp/*_x.npy and ../tmp/*_m.npy
#===============================================================================

if os.path.exists('../data/'):
	data_files = os.listdir('../data/')
else:
	print 'ERROR: data directory not found!'

if not os.path.exists('../tmp/'):
	os.makedirs('../tmp/')

inputs = []
labels = []
batch_size = 64

bar = pgb(len(data_files), max_width=50)
print 'starting processing..'
for i in range(len(data_files)):
	bar.numerator = i+1
	clip_name = data_files[i][:-4]
	data_x = np.load('../data/'+data_files[i])
	anchors = np.random.randint(data_x.shape[0]*.6, size=batch_size-1)
	X = (data_x.reshape(1, data_x.shape[0], 1), )
	M = (np.ones([1, data_x.shape[0], 1], dtype='uint8'), )
	for anchor in anchors:
		x = np.append(data_x[anchor:], np.zeros([anchor], dtype='uint8')).reshape(1, data_x.shape[0], 1)
		m = np.concatenate((np.ones([1, data_x.shape[0]-anchor, 1], dtype='uint8'), np.zeros([1, anchor, 1], dtype='uint8')), axis=1)
		X+= (x, )
		M+= (m, )
	X = np.concatenate(X)
	M = np.concatenate(M)
	np.save('../tmp/'+clip_name+'x.npy', X)
	np.save('../tmp/'+clip_name+'m.npy', M)
	print '\033[Ffiles processed:', bar

print 'inputs and masks stored in ../tmp/ '
