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

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten
from keras import metrics
from sklearn.metrics import classification_report

import arxiv

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

embedding_layer = arxiv.build_embedding_layer(
    maxlen=maxlen, batch_size=batch_size)
gc.collect()

print("setting up model layout...")
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(maxlen*2, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(maxlen*2)))
model.add(Dropout(0.2))
model.add(Dense(n_classes, activation='softmax'))
# try using different optimizers and different optimizer configs?
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",  # sgd ?
              metrics=[metrics.sparse_categorical_accuracy])
# summarize the model
print('Training model...')
print(model.summary())

model.fit(x_train, y_train,
          batch_size=batch_size,
          # increase epochs to 10-50 when we're at reasonable initial performance (0.8 or higher)
          epochs=2,
          validation_split=0.2)

# evaluate the model
print("Evaluating model on test data...")
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
print("Saving model to disk...")
model.save("model-2x300-c10.h5")

print("Per-class test measures:")
y_pred = model.predict_classes(x_test, batch_size=1)
print(classification_report(y_test, y_pred))
