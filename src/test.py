from __future__ import print_function
import numpy as np
import gc
import json
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, TimeDistributed
from keras import metrics
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import Embedding, Input
from keras.models import load_model
import arxiv

gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=0.4, allow_growth=True)
config = tf.ConfigProto(intra_op_parallelism_threads=16,
                        inter_op_parallelism_threads=16, allow_soft_placement=True, gpu_options=gpu_options)

session = tf.Session(config=config)
K.set_session(session)


maxlen = 480
# embedding_layer = arxiv.build_embedding_layer(maxlen=maxlen)

model = load_model('model.h5')

print('Loading data...')
x_train, x_test, y_train, y_test = arxiv.load_data(maxlen=None, start_char=None, num_words=1_000_000,
                                                   shuffle=False, setup_labels='f1-envs', full_data=False, max_per_class=5_000)

y_prob = model.predict(x_test, batch_size=256, verbose=1)
y_pred = y_prob.argmax(axis=-1)

print("y_prob: ", y_prob)
print("y_pred: ", y_pred)
