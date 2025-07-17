import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

#print(tfds.list_builders())

current_dir = os.path.dirname(os.path.abspath(__file__))

# FOR SIFT1M DATASET
sift1m_database = tfds.load("sift1m", split='database', data_dir=current_dir)
sift1m_test = tfds.load("sift1m", split='test', data_dir=current_dir)

deep1b_database = tfds.load("deep1b", split='database', data_dir=current_dir)
deep1b_test = tfds.load("deep1b", split='test', data_dir=current_dir)