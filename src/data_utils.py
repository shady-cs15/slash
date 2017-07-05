#===============================================================================
# MIT License
# Copyright (c) 2017 Satyaki Chakraborty

# load_file: read .wav file, normalizes it,
# returns quantized wave form
# save_file: takes quantized wave as input,
# rescales it, writes into .wav file
#===============================================================================

import librosa
import random
import numpy as np
import matplotlib.pyplot as plt

def mu_law_encoding(c_wave, n_channels=256):
	mu = n_channels-1
	abs_wave = np.minimum(np.abs(c_wave), 1.)
	magnitude = np.log(1. + mu * abs_wave)/np.log(1. + mu)
	c_wave = np.sign(c_wave)*magnitude
	return ((c_wave+1)/2*mu + 0.5).astype('uint8')

def mu_law_decoding(d_wave, n_channels=256):
	mu = n_channels-1
	d_wave = d_wave.astype('float32')
	d_wave = 2 * (d_wave/mu) - 1
	magnitude = (1./mu) * ((1. + mu)**abs(d_wave) - 1.)
	return np.sign(d_wave) * magnitude

def load_file(file_name, sr, q_levels):
	raw, sr = librosa.load(file_name, mono=False, sr=sr)
	wave = raw[1]
	scale = np.max(np.abs(wave))
	wave /= scale
	wave = mu_law_encoding(wave, q_levels)
	return wave

def save_file(file_name, wave, sr, q_levels):
	wave = mu_law_decoding(wave, q_levels)
	scale = np.max(np.abs(wave))
	wave /= scale
	librosa.output.write_wav(file_name, wave, sr)

def add_noise_and_augment(data, prob=0.33, n_noisy_augs=4):
	pass
