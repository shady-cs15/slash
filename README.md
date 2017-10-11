# Slash
- Teaching neural networks how to play guitar
- Based on SampleRNN (Mehri et al. 2016)
- Under progress!!

![alt](https://shady-cs15.github.io/blogs/images/slash/samp-rnn.gif)

# License
MIT License

# Results (generated over time)
[link to video](https://www.youtube.com/watch?v=wYrbC7KOuNw)

# Preprocessing .wav files
* navigate to project directory
* create `waves` directory inside the current directory
* copy all the wave files inside this directory
* run `$ python preprocess.py`
* A `data.npy` file is created inside `tmp` directory

# Training model
* run `$ python train_model.py`
* weights are stored in `../params/` directory
* logs are stored in `../logs/` directory
* generated samples are stored in `../gen/` directory

# Tensorboard visualization
* training logs dir: `../logs/train/`
* validation logs dir: `../logs/valid/`

# Generating samples
* samples are automatically generated after `gen_freq` iterations
* samples are generated inside `gen` directory

# Changing model/training configurations
* Just modify the corresponding parameters in `settings.cfg`

# Blog
[link to blog](https://shady-cs15.github.io/blogs/slash.html)


