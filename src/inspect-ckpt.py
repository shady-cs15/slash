import os
import random

files = os.listdir('../tmp/')
files.remove('.floyddata')
files = [(('/input/'+files[i+1]), ('/input/'+files[i]))  for i in range(0, len(files), 2)]
random.seed(3)
random.shuffle(files)
print files[-5:]