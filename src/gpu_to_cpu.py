import warnings
import numpy as np
import gc
import json
import h5py

import tensorflow as tf
from keras.models import load_model
from keras.utils import Sequence
from keras.utils.data_utils import get_file
from keras.preprocessing import sequence
from keras.preprocessing.sequence import _remove_long_seq
from keras.models import Sequential
from keras.layers import Embedding, Input, Dense, CuDNNLSTM, Bidirectional, LSTM
from keras import metrics
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


def load_vocab():
    with open('data/word_index.json') as json_data:
        return json.load(json_data)


def load_index_vocab():
    index_vocab = {}
    with open('data/word_index.json') as json_data:
        for word, index in json.load(json_data).items():
            index_vocab[index] = word
    return index_vocab


def load_glove():
    glove = {}
    with open('data/glove.model.txt') as glove_data:
        for line in glove_data:
            items = line.split()
            key = items[0]
            glove[key] = np.asarray(items[1:], dtype='float32')
    return glove


def build_embedding_layer(with_input=False, maxlen=480, vocab_dim=300, mask_zero=True):
    print("-- loading word embeddings, this may take a couple of minutes...")
    index_dict = load_vocab()
    word_vectors = load_glove()
    # adding 1 to account for 0th index (for masking)
    n_symbols = len(index_dict) + 1
    print("-- known dictionary items: ", n_symbols)
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]
    print("-- embeddings ")
    if not with_input:
        embedding_layer = Embedding(
            mask_zero=mask_zero,
            output_dim=vocab_dim, input_dim=n_symbols, input_length=maxlen, trainable=False, weights=[embedding_weights])
        return embedding_layer
    else:
        # define inputs here
        input_1 = Input(shape=(maxlen,), dtype='int32')
        embedding_layer = Embedding(
            weights=[embedding_weights],
            mask_zero=mask_zero,
            output_dim=vocab_dim, input_dim=n_symbols, input_length=maxlen, trainable=False)(input_1)
        return (embedding_layer, input_1)


layer_size = 128
batch_size = 128
n_classes = 13
maxlen = 480

gpu_model = load_model(
    "models/confusion_bilstm128_batch128_cat13_gpu_notebook.h5")
gpu_model.save_weights('GPU.weights')

embedding_layer = build_embedding_layer(maxlen=maxlen, mask_zero=False)
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

cpu_model.save("13_class_statement_classification_bilstm.h5")
