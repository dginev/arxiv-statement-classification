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
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import arxiv

maxlen = 128  # 5 sentences of 25 words each? too small?
batch_size = 32  # 128. what is the optimum here?

print('Loading data...')
(x_train, y_train), (x_test, y_test) = arxiv.load_data(maxlen=maxlen)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
gc.collect()

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

y_train = np.array(y_train)
y_test = np.array(y_test)

index_dict = arxiv.load_vocab()
word_vectors = arxiv.load_glove()
vocab_dim = 300
# adding 1 to account for 0th index (for masking)
n_symbols = len(index_dict) + 1
embedding_weights = np.zeros((n_symbols, vocab_dim))
for word, index in index_dict.items():
    embedding_weights[index, :] = word_vectors[word]

# define inputs here
embedding_layer = Embedding(
    mask_zero=True,
    output_dim=vocab_dim, input_dim=n_symbols, input_length=maxlen, trainable=True)
# if you don't do this, the next step won't work
embedding_layer.build((None,))
embedding_layer.set_weights([embedding_weights])

model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(300, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['acc'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          validation_split=0.2)

# evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk")
