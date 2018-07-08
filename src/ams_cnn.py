'''Trains a deep CNN on the arXiv AMS environment classification task.

adapted from a tutorials:
https://towardsdatascience.com/another-twitter-sentiment-analysis-with-python-part-11-cnn-word2vec-41f5e28eda74
https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Conv1D, GlobalMaxPooling1D
from keras import metrics
from keras import backend as K
from sklearn.metrics import classification_report
from keras.layers.merge import concatenate

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
strict_labels = 'f1-envs'
n_classes = 23  # ams classes/labels (0-22)
if strict_labels:  # down to (0-10) if strict
    if strict_labels == "envs-only":
        n_classes = 22
    elif strict_labels == "strict-envs-only":
        n_classes = 10
    elif strict_labels == "f1-envs":
        n_classes = 9
    else:
        n_classes = 11

print('Loading data...')
(x_train, y_train), (x_test, y_test) = arxiv.load_data(
    maxlen=maxlen, strict_labels=strict_labels, full_data=False)
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

glove_dim = 300  # GloVe vector dimensions
nb_feature_maps = int(maxlen/2)

(embedding_layer, input_1) = arxiv.build_embedding_layer(with_input=True,
                                                         maxlen=maxlen, batch_size=batch_size, mask_zero=False)
gc.collect()

ngram_filters = [2, 3, 4, 5, 6, 8]
conv_filters = []

for n_gram in ngram_filters:
    tower_i = Conv1D(filters=nb_feature_maps,
                     kernel_size=n_gram, activation='relu')(embedding_layer)
    tower_i = GlobalMaxPooling1D()(tower_i)
    conv_filters.append(tower_i)

print("setting up model layout...")
output = concatenate(conv_filters)
output = Dropout(0.5)(output)
output = Dense(nb_feature_maps * len(ngram_filters), activation="relu")(output)
output = Dropout(0.2)(output)
output = Dense(n_classes, activation='softmax')(output)
# summarize the model
model = Model(inputs=[input_1], outputs=output)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=[metrics.sparse_categorical_accuracy])

print('Training model...')
print(model.summary())

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_split=0.2)


# evaluate the model
print("Evaluating model on test data...")
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
print("Saving model to disk...")
model.save("model-cnn-6gram-f1-9-classes.h5")

# print("Per-class test measures:")
# y_pred = model.predict_classes(x_test, verbose=1)
# print(classification_report(y_test, y_pred))
