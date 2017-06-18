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
import sys
from etaprogress.progress import ProgressBar as pgb
from ConfigParser import ConfigParser as Config

load_path = '../waves2/'
save_path = '../tmp/'
if not os.path.exists(load_path):
    print 'ERROR: waves directory not found!'
    sys.exit(1)
else:
    wave_files = os.listdir(load_path)
    wave_files = sorted(wave_files)

if not os.path.exists(save_path):
	os.makedirs(save_path)

cfg = Config()
cfg.read('settings.cfg')
sr = int(cfg.get('process', 'bitrate'))
q_levels = int(cfg.get('process', 'q_levels'))
seq_len = int(cfg.get('process', 'seq_len'))

bar = pgb(len(wave_files), max_width=50)
print 'starting preprocessing..'

unit = int(1e4)
max_len = seq_len*unit
stride = 2*unit
data = ()
for i in range(len(wave_files)):
    bar.numerator = i+1
    clip_name = wave_files[i][:-4]
    q_wave = du.load_file(load_path+wave_files[i], sr, q_levels)
    length = (q_wave.shape[0]/unit)*unit
    q_wave = q_wave[:length]
    start_ptr = 0
    while(start_ptr +stride<length):
        subclip = q_wave[start_ptr:start_ptr+max_len]
        start_ptr+=stride
        if subclip.shape[0]<max_len:
            continue
        subclip = (subclip).reshape(1, max_len)
        data += (subclip,)
    print '\033[Ffiles processed:', bar
data = np.concatenate(data).astype(np.uint8)
save_file = save_path+'data.npy'
np.save(save_file, data)
print 'quantized waves stored in', save_file
print 'data shape:', data.shape
