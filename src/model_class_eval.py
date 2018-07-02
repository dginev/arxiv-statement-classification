'''Trains a Bidirectional LSTM on the arXiv AMS environment classification task.

adapted from the official Keras examples:
https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py
'''

# of all the weird dependency hells...
# needed exactly cuda 9.0 and libcudnn 7, latter obtainable via:
# http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/
# the former via (and you need the 3 updates for 9.0)
# https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=deblocal

from __future__ import print_function
import numpy as np
import gc
import json

import tensorflow as tf
from keras.preprocessing import sequence
import keras.models
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten
from keras import metrics
from keras import backend as K
from sklearn.metrics import classification_report

import arxiv

# Use full CPU capacity
config = tf.ConfigProto(intra_op_parallelism_threads=16,
                        inter_op_parallelism_threads=16, allow_soft_placement=True)
session = tf.Session(config=config)
K.set_session(session)

maxlen = 150  # sentences of 25 words each? Also compare "15", 50" vs "250", "500" word window extremes
# what is the optimum here? the average arXiv document seems to have 110 paragraphs ?!
batch_size = 128
strict_labels = True
n_classes = 23  # ams classes/labels (0-22)
if strict_labels:  # down to (0-10) if strict
    n_classes = 11

print('Loading data...')
(x_train, y_train), (x_test, y_test) = arxiv.load_data(
    maxlen=maxlen, strict_labels=strict_labels)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
gc.collect()

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post')

y_train = np.array(y_train)
y_test = np.array(y_test)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Loading...')
model = keras.models.load_model("models/model-2x300-c10.h5")
print(model.summary())

print("Per-class test measures:")
y_pred = model.predict_classes(x_test, verbose=1)
print(classification_report(y_test, y_pred))
