#===============================================================================
# Pre processing script for wave quantization
# segmentation and mask generation
# Reads raw waves from ../waves/*.wav files
# Generates mu law encoded quantizations
# and correspodning masks
# Stores them in ../tmp/data.npy
# data is of shape nsubclips x 2 x max_len
#===============================================================================

import numpy as np
import data_utils as du
import os
from etaprogress.progress import ProgressBar as pgb
import sys


if not os.path.exists('../waves/'):
    print 'ERROR: waves directory not found!'
    sys.exit(1)
else:
    wave_files = os.listdir('../waves/')
    wave_files = sorted(wave_files)

if not os.path.exists('../tmp/'):
	os.makedirs('../tmp/')

bar = pgb(len(wave_files), max_width=50)
print 'starting preprocessing..'

unit = int(1e4)
max_len = 8*unit
stride = 2*unit
data = ()
for i in range(len(wave_files)):
    bar.numerator = i+1
    clip_name = wave_files[i][:-4]
    q_wave = du.load_file('../waves/'+wave_files[i])
    length = (q_wave.shape[0]/unit)*unit
    q_wave = q_wave[:length]
    start_ptr = 0
    while(start_ptr +stride<length):
        subclip = q_wave[start_ptr:start_ptr+max_len]
        mask = np.ones(subclip.shape[0])
        if subclip.shape[0]<max_len:
            subclip = np.concatenate([subclip, np.zeros([max_len-subclip.shape[0]])])
            mask = np.concatenate([mask, np.zeros(max_len-mask.shape[0])])
        subcliptuple = np.array([subclip, mask]).reshape(1, 2, max_len)
        data += (subcliptuple,)
        start_ptr+=stride
    print '\033[Ffiles processed:', bar
data = np.concatenate(data).astype(np.uint8)
np.save('../tmp/data.npy', data)
print 'waves and masks stored in ../tmp/data.npy'
