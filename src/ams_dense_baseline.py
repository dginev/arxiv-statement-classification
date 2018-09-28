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
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten
from keras import metrics
from keras import backend as K
from sklearn.metrics import classification_report

import arxiv

# Use full CPU capacity
gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=0.333, allow_growth=True)
config = tf.ConfigProto(intra_op_parallelism_threads=16,
                        inter_op_parallelism_threads=16, allow_soft_placement=True, gpu_options=gpu_options)

session = tf.Session(config=config)
K.set_session(session)

# Analyzing the arxiv dataset seems to indicate a maxlen of 300 is needed to fit 99.2% of the data
#                                               a maxlen of 150 fits 94.03%, and a maxlen of 600 covers 99.91% of paragraphs
maxlen = 300  # sentences of 25 words each? Also compare "15", 50" vs "250", "500" word window extremes
# what is the optimum here? the average arXiv document seems to have 110 paragraphs ?!
batch_size = 128  # 32, 64, 128

# Results on arxiv 08.2018 demo_ams.npz
#                      precision    recall  f1-score   support
# 23 classes: avg / total   0.33      0.32      0.30     91727
# 9 classes:                0.67      0.70      0.66     91727

setup_labels = 'f1-envs'  # False
classes_for_label = {
    "no-other": 22,
    "strict-envs": 11,
    "stricter-envs": 10,
    "f1-envs": 9,
    "definition-binary": 2
}
n_classes = 23  # ams classes/labels (0-22)
if setup_labels and setup_labels in classes_for_label:
    n_classes = classes_for_label[setup_labels]

print('Loading data...')
(x_train, y_train), (x_test, y_test) = arxiv.load_data(
    maxlen=maxlen, setup_labels=setup_labels, full_data=False)
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
    maxlen=maxlen, batch_size=batch_size, mask_zero=False)
gc.collect()

print("setting up model layout...")
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(maxlen, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
# try using different optimizers and different optimizer configs?
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=[metrics.sparse_categorical_accuracy])
# summarize the model
print('Training model...')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_split=0.2)

print('Model summary:')
print(model.summary())

# evaluate the model
print("Evaluating model on test data...")
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
print("Saving model to disk...")
model.save("model-300-baseline-f1-9-classes-big.h5")

print("Per-class test measures:")
y_pred = model.predict_classes(x_test, verbose=1)
print(classification_report(y_test, y_pred))
