'''Trains a  simple MPL baseline model for AMS classification
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
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten
from keras import metrics
from keras import backend as K
from sklearn.metrics import classification_report

import arxiv

# Try to use full CPU capacity, and a 1/3 of the GPU memory.
# the CPU options don't seem to be working however, at least I only see a single thread used... missing something here.
# optimized for my 1080 Ti and Threadripper 1950x
gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=0.333, allow_growth=True)
config = tf.ConfigProto(intra_op_parallelism_threads=16,
                        inter_op_parallelism_threads=16, allow_soft_placement=True, gpu_options=gpu_options)
session = tf.Session(config=config)
K.set_session(session)

# Analyzing the arxiv dataset seems to indicate a maxlen of 300 words is needed to fit 99.2% of the paragraphs
#                                               a maxlen of 150 fits 94.03%, and a maxlen of 600 covers 99.91% of paragraphs
# what is the optimum here? also, the average arXiv document seems to have 110 paragraphs ?!
# maxlen = 300
# update: we limit the maxlen when we generate the dataset, to 480, see `max_words` in ams_tar_to_hdf5.py

setup_labels = "confusion-envs-v3"
n_classes = 8

# We can't enable shuffling unless we restrict the max_per_class to a much smaller number.
# Shuffling should essentially be a minor issue given the amount of data presented to the model
x_train, x_test, y_train, y_test = arxiv.load_data(
    maxlen=None, setup_labels=setup_labels, start_char=None, full_data=False, shuffle=False, num_words=1_000_000, max_per_class=900_000
)
maxlen = 480  # don't pass into laod_data for now, speedier to assume it's preset to 480
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
gc.collect()

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

y_train = np.array(y_train)
y_test = np.array(y_test)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

embedding_layer = arxiv.build_embedding_layer(
    maxlen=maxlen, mask_zero=False)
gc.collect()

use_dropout = True  # disable if running on small dataset demo
print("-- setting up model layout...")
model = Sequential()
model.add(embedding_layer)
if use_dropout:
    model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(maxlen, activation='relu'))
model.add(Dense(maxlen, activation='relu'))
if use_dropout:
    model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))
# try using different optimizers and different optimizer configs?
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=[metrics.sparse_categorical_accuracy])
# summarize the model
print('-- training model...')

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_split=0.2)

print('Model summary:')
print(model.summary())

# evaluate the model
print("-- evaluating model on test data...")
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
print("-- saving model to disk...")
# e.g. rename from -big to -demo, for small data
model.save("model-mlp-baseline-big.h5")

print("Per-class test measures:")
y_pred = model.predict_classes(x_test, verbose=1)
print(classification_report(y_test, y_pred))

# -- scratch --
# First results on arxiv 08.2018 demo_ams.npz (ams paragraphs v2)
#                      precision    recall  f1-score   support
# 23 classes: avg / total   0.34      0.33      0.31     91727 (50k)
#                           0.23      0.22      0.22     15065 (5k)
#
# 9 classes:                0.67      0.71      0.67     91727 (50k)
#                           0.59      0.65      0.59     15065 (5k)
