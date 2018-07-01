# -*- coding: utf-8 -*-
"""arXiv topic classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import _remove_long_seq
import numpy as np
import json
import warnings
import gc
from keras.layers import Embedding


def load_data(path='data/demo_ams.npz', num_words=200_000, skip_top=0,
              maxlen=None, test_split=0.2, seed=521,
              start_char=1, oov_char=2, index_from=2, strict_labels=False, **kwargs):
    """Loads the Reuters newswire classification dataset.

    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).
        num_words: max number of words to include. Words are ranked
            by how often they occur (in the training set) and only
            the most frequent words are kept
        skip_top: skip the top N most frequently occurring words
            (which may not be informative).
        maxlen: truncate sequences after this length.
        test_split: Fraction of the dataset to be used as test data.
        seed: random seed for sample shuffling.
        start_char: The start of a sequence will be marked with this character.
            Set to 1 because 0 is usually the padding character.
        oov_char: words that were cut out because of the `num_words`
            or `skip_top` limit will be replaced with this character.
        index_from: index actual words with this index and higher.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    Note that the 'out of vocabulary' character is only used for
    words that were present in the training set but are not included
    because they're not making the `num_words` cut here.
    Words that were not seen in the training set but are in the test set
    have simply been skipped.
    """
    # TODO: Enhance when remote download is possible
    # path = get_file(path, origin='https://s3.amazonaws.com/text-datasets/reuters.npz',
    #                 file_hash='87aedbeb0cb229e378797a632c1997b6')

    with np.load(path) as f:
        xs, labels = f['x'], f['y']

    # TODO: BAD!!! If we are sampling a small amount from the data, it needs to preserve the distribution.
    #       At least grab a max per category.

    # A "Zero Rule" classifier with this restriction will have accuracy of 0.077725805
    max_per_class = 50_000
    print("reducing data to ", max_per_class, " per class...")
    selection_counter = {}
    xs_reduced = []
    labels_reduced = []

    for (idx, label) in enumerate(labels):
        if not(label in selection_counter):
            selection_counter[label] = 0
        if selection_counter[label] < max_per_class:
            selection_counter[label] += 1
            xs_reduced.append(xs[idx])
            labels_reduced.append(label)
    xs = np.array(xs_reduced)
    labels = np.array(labels_reduced)
    gc.collect()

    print("shuffling data...")
    np.random.seed(seed)
    indices = np.arange(len(xs))
    np.random.shuffle(indices)

    xs = xs[indices]
    labels = labels[indices]

    if strict_labels:
        print("Reducing to 10 strict labels, starting at 1")
        # strict_dict = {
        #     "definition": 1,
        #     "example": 2,
        #     "notation": 3,
        #     "problem": 4,
        #     "proof": 5,
        #     "proposition": 6,
        #     "question": 7,
        #     "remark": 8,
        #     "theorem": 9,
        #     "other": 10
        # }
        strict_map = {
            0: 10,
            1: 1,
            2: 6,
            3: 10,
            4: 6,
            5: 6,
            6: 6,
            7: 6,
            8: 1,
            9: 2,
            10: 6,
            11: 9,
            12: 3,
            13: 10,
            14: 10,
            15: 4,
            16: 5,
            17: 6,
            18: 7,
            19: 8,
            20: 8,
            21: 8,
            22: 9
        }
        labels = np.array([int(strict_map[l]) for l in labels])

    # Might as well report a summary of what is in the labels...
    label_summary = dict.fromkeys(range(0, 23), 0)
    for label in labels:
        label_summary[label] += 1
    print("Label summary: ", label_summary)

    print("preparing sets...")
    print("- start char")
    if start_char is not None:
        xs_len = len(xs)
        iterations = 0
        while iterations < xs_len:
            xs[iterations] = [start_char] + \
                [w + index_from for w in xs[iterations]]
            iterations += 1
    elif index_from:
        xs_len = int(len(xs))
        iterations = 0
        while iterations < xs_len:
            xs[iterations] = [w + index_from for w in xs[iterations]]
            iterations += 1

    print("- maxlen")
    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
    print("- num words")
    if not num_words:
        num_words = max([max(x) for x in xs])

    print("- oov char")
    # by convention, use -1 as OOV word
    # reserve 'index_from' (=2 by default, as the para index starts at 1) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    # 3 is the most common word ('the')
    if oov_char is not None:
        xs_len = int(len(xs))
        iterations = 0
        while iterations < xs_len:
            xs[iterations] = [w if skip_top <= w <
                              num_words else oov_char for w in xs[iterations]]
            iterations += 1
    else:
        xs_len = int(len(xs))
        iterations = 0
        while iterations < xs_len:
            xs[iterations] = [w for w in xs[iterations]
                              if skip_top <= w < num_words]
            iterations += 1

    idx = int(len(xs) * (1 - test_split))
    print("performing train/test cutoff at index ", idx, "/", len(xs), '...')

    x_train, x_test = np.array(xs[:idx]), np.array(xs[idx:])
    y_train, y_test = np.array(labels[:idx]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)


def get_word_index(path='data/arxiv_word_index.json'):
    """Retrieves the dictionary mapping word indices back to words.

    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).

    # Returns
        The word index dictionary.
    """
    # path = get_file(path,
    #                 origin='https://s3.amazonaws.com/text-datasets/reuters_word_index.json',
    #                 file_hash='4d44cc38712099c9e383dc6e5f11a921')
    f = open(path)
    data = json.load(f)
    f.close()
    return data


def load_vocab():
    index_dict = {}
    with open('data/arxiv_word_index.json') as json_data:
        index_dict = json.load(json_data)
    return index_dict


def load_glove():
    glove = {}
    with open('data/glove.arxmliv.5B.300d.txt') as glove_data:
        for line in glove_data:
            items = line.split()
            key = items.pop(0)
            glove[key] = [float(item) for item in items]
    return glove


def build_embedding_layer(maxlen=256, index_from=2, vocab_dim=300, batch_size=50):
    print("loading word embeddings...")
    index_dict = load_vocab()
    word_vectors = load_glove()
    # adding 2 to account for 0th index (for masking), as well as 1st for start and 2nd for oov
    # most frequent word ('the') hence has index 3 in the loaded data, so we start loading the embeddings from index 3 as well
    n_symbols = len(index_dict) + index_from + 1
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index+index_from, :] = word_vectors[word]

    # define inputs here
    embedding_layer = Embedding(
        mask_zero=True,
        output_dim=vocab_dim, input_dim=n_symbols, input_length=maxlen, trainable=False)
    # if you don't do this, the next step won't work
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_weights])
    return embedding_layer
