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
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten
from keras import metrics
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

import arxiv

# Use full CPU capacity
# gpu_options = tf.GPUOptions(
#     per_process_gpu_memory_fraction=0.333, allow_growth=True)
config = tf.ConfigProto(intra_op_parallelism_threads=16,
                        inter_op_parallelism_threads=16, allow_soft_placement=True)  # , gpu_options=gpu_options)

session = tf.Session(config=config)
K.set_session(session)

# 08.2017 (mathformula replacements)
# Analyzing the arxiv dataset seems to indicate a maxlen of 300 is needed to fit 99.2% of the data
#                                               a maxlen of 150 fits 94.03%, and a maxlen of 600 covers 99.91% of paragraphs
# 08.2018 (subformula lexemes)
# Analyzing the arxiv dataset seems to indicate a maxlen of 960 is needed to fit 99.2% of the data
#                                               a maxlen of 480 fits 96.03%, and a maxlen of 300 covers 90.0% of paragraphs

maxlen = 480

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
x_train, x_test, y_train, y_test = arxiv.load_data(maxlen=maxlen,
                                                   setup_labels=setup_labels, full_data=False, max_per_class=50_000)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
gc.collect()

y_train = np.array(y_train)
y_test = np.array(y_test)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)

embedding_layer = arxiv.build_embedding_layer(maxlen=maxlen)
gc.collect()

print("setting up model layout...")
model_file = "bilstm-120-dual-9cat-bigbatch"

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(int(maxlen/4), return_sequences=True)))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(int(maxlen/4))))
model.add(Dropout(0.1))
model.add(Dense(n_classes, activation='softmax'))
# try using different optimizers and different optimizer configs?
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              weighted_metrics=[metrics.sparse_categorical_accuracy])
# summarize the model
print('Training model...')
print(model.summary())

# Keep only a single checkpoint, the best over test accuracy.
checkpoint = ModelCheckpoint(model_file+"-checkpoint.h5",
                             monitor='val_weighted_sparse_categorical_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

model.fit(x_train, y_train,
          # what is the optimum here? the average arXiv document seems to have 110 paragraphs ?!
          batch_size=256,  # 32, 64, 128
          # Classifies into: acknowledgement(0), algorithm(1), caption(2), proof(3), assumption(4), definition(5), problem(6), remark(7), other(8)
          # f1-envs only, based on ratios in full dataset
          # https://docs.google.com/spreadsheets/d/16I9969_QcU4J9EtglGKZpLHVeNcFIeDGNU4trhi53Vc/edit#gid=1538283102
          #   class_weight={0: 2500, 1: 1000, 2: 12500, 3: 2.6,
          #                 4: 450, 5: 17, 6: 400, 7: 17, 8: 0.5},
          class_weight=class_weights,
          epochs=10,
          verbose=1,
          callbacks=[checkpoint],
          validation_split=0.2)

# evaluate the model -- redundant with per-class tests
# print("Evaluating model on test data...")
# scores = model.evaluate(x_test, y_test, verbose=1, batch_size=256)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
print("Saving model to disk : %s " % model_file)
model.save(model_file+'.h5')

print("Per-class test measures:")
y_pred = model.predict_classes(x_test, verbose=1, batch_size=256)
print(classification_report(y_test, y_pred))
