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
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten
from keras import metrics

import arxiv

maxlen = 150  # 5? 10? sentences of 25 words each?
# what is the optimum here? the average arXiv document seems to have 110 paragraphs ?!
batch_size = 100
strict_labels = False
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
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

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
model.add(Bidirectional(LSTM(75)))
model.add(Dropout(0.25))
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
          epochs=10,
          validation_split=0.2)

# evaluate the model
print("Evaluating model on test data...")
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
print("Saving model to disk...")
model.save("model-64.h5")
# model_json = model.to_json()
# with open("model-64.json", "w") as json_file:
#     json_file.write(model_json)
print("Saved model to disk. Done!")

#
# LSTM(32) (sgd optimizer, metric sparse_categorical_accuracy) run:
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 150, 300)          224002800
# _________________________________________________________________
# bidirectional_1 (Bidirection (None, 64)                85248
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 64)                0
# _________________________________________________________________
# dense_1 (Dense)              (None, 23)                1495
# =================================================================
# Total params: 224,089,543
# Trainable params: 86,743
#
# Train on 411703 samples, validate on 102926 samples
# Epoch 1/2
# 411703/411703 [==============================] - 2690s 7ms/step - loss: 2.3600 - sparse_categorical_accuracy: 0.2427
#                                                             - val_loss: 2.1382 - val_sparse_categorical_accuracy: 0.3077
# Epoch 2/2
# 411703/411703 [==============================] - 2702s 7ms/step - loss: 2.0567 - sparse_categorical_accuracy: 0.3320 - val_loss: 1.9705 - val_sparse_categorical_accuracy: 0.3538
# Evaluating model on test data...
# sparse_categorical_accuracy: 35.39%
# Saving model to disk...
# Saved model to disk. Done!
