import numpy as np
import gc
import json

import keras
import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, Bidirectional, LSTM
from keras.utils.np_utils import to_categorical
from keras import metrics
import arxiv

layer_size = 128
batch_size = 256
n_classes = 8
maxlen = 480

gpu_model = load_model("models/v3_bilstm128_batch256_cat8_gpu.h5")
gpu_model.save_weights('GPU.weights')

embedding_layer = arxiv.build_embedding_layer(maxlen=maxlen, mask_zero=False)
gc.collect()

cpu_model = Sequential()
cpu_model.add(embedding_layer)
cpu_model.add(Bidirectional(LSTM(layer_size, return_sequences=True)))
cpu_model.add(Bidirectional(LSTM(layer_size // 2, return_sequences=True)))
cpu_model.add(LSTM(layer_size // 2))
cpu_model.add(Dense(n_classes, activation='softmax'))

cpu_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="adam",
                  weighted_metrics=[metrics.sparse_categorical_accuracy])
cpu_model.load_weights('GPU.weights')

# Print model summary
print(cpu_model.summary())

cpu_model.save("cpu_model.h5")
