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
from keras.layers import Embedding, Input


def load_data(path='data/demo_ams.npz', num_words=200_000, skip_top=0,
              maxlen=None, test_split=0.2, seed=521,
              start_char=1, oov_char=2, index_from=2, setup_labels=False, full_data=False, **kwargs):
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

    if full_data:
        path = "data/arxiv_ams.npz"
    with np.load(path) as f:
        xs, labels = f['x'], f['y']

    # TODO: BAD!!! If we are sampling a small amount from the data, it needs to preserve the distribution.
    #       At least grab a max per category.

    max_per_class = 50_000  # Leads to 659_216 total expressions
    # A "Zero Rule" classifier with this 50,000 restriction will have accuracy of 0.076
    #               if strict classes are on, the zero rule accuracy would be 0.29
    #               with the f1-based 9 classes, zero rule wouild be a very high 0.667

    if full_data:
        # A "Zero Rule" classifier with this restriction and f1-env will have accuracy of 0.62
        #   total paragraphs would be 14,740,530, with Other numbered at 9,137,806
        max_per_class = 500_000

    # Also drops empty rows, and rows with NaN, just in case
    print("reducing data to ", max_per_class, " per class...")
    selection_counter = {}
    xs_reduced = []
    labels_reduced = []

    for (idx, label) in enumerate(labels):
        if not(label in selection_counter):
            selection_counter[label] = 0
        if selection_counter[label] < max_per_class:
            x = xs[idx]
            if len(x) > 0 and not(np.isnan(np.min(x))):
                selection_counter[label] += 1
                xs_reduced.append(x)
                labels_reduced.append(label)
    xs = np.array(xs_reduced)
    labels = np.array(labels_reduced)
    gc.collect()

    if setup_labels:
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
        other_label = 13
        if setup_labels == "stricter-envs":
            print("Reducing to 9 label classes")
            stricter_map = {
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
                11: 10,
                12: 10,
                13: 10,
                14: 10,
                15: 4,
                16: 5,
                17: 6,
                18: 10,
                19: 8,
                20: 8,
                21: 8,
                22: 10
            }
            other_label = 10
            labels = np.array([int(stricter_map[l]) for l in labels])
        elif setup_labels == "f1-envs":
            # (as evaluated on a 2 layer biLSTM(150)+biLSTM(150))
            # Based on experimental f1-score on the full 23 classes where:
            #
            # >0.5 assumption 2, definition 8, example 9, notation 12, problem 15, remark 19
            # >0.7 algorithm 1, caption 3, proof 16,
            # >0.8
            # >0.9 acknowledgement 0
            # Rearrange label indices in decreasing expected f1
            #
            # Down to 10 "high" performing label classes
            # First attempt, 68% f1
            # whitelist = {0: 0, 1: 1, 3: 2, 16: 3,
            #              2: 4, 8: 5, 9: 6, 12: 7, 15: 8, 19: 9}
            #
            # Second attempt, drop worst (notation), % f1:
            # whitelist = {0: 0, 1: 1, 3: 2, 16: 3,
            #              2: 4, 8: 5, 9: 6, 15: 7, 19: 8}
            #
            # Third attempt, also drop next worst (example), % f1:
            whitelist = {0: 0, 1: 1, 3: 2, 16: 3,
                         2: 4, 8: 5, 15: 6, 19: 7}

            xs_env = []
            labels_env = []
            other_label = len(whitelist)
            print("using f1-based label whitelist of size ", len(whitelist)+1)
            for (idx, label) in enumerate(labels):
                xs_env.append(xs[idx])
                if label in whitelist:
                    labels_env.append(whitelist[label])
                else:
                    labels_env.append(other_label)
            xs = np.array(xs_env)
            labels = np.array(labels_env)
        elif setup_labels == "strict-envs":
            strict_envs = {
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
            print("Reducing to 10 label classes")
            other_label = 10
            labels = np.array([int(strict_envs[l]) for l in labels])
        elif setup_labels == "definition-binary":
            print("Reducing to 2 label classes")
            other_label = 1
            labels = np.array([0 if l == 8 else 1 for l in labels])
        # dropping requested categories ("no-*")
        if setup_labels == "no-other" or setup_labels == "stricter-envs":
            print("ignoring Other category from dataset")
            xs_env = []
            labels_env = []
            for (idx, label) in enumerate(labels):
                if label != other_label:
                    if label > other_label:
                        # the labels need to be in a tight integer sequence, for training to work smoothly, so close the gap we open by removing Other
                        label = label-1
                    xs_env.append(xs[idx])
                    labels_env.append(label)
            xs = np.array(xs_env)
            labels = np.array(labels_env)
        gc.collect()

    print("shuffling data...")
    np.random.seed(seed)
    indices = np.arange(len(xs))
    np.random.shuffle(indices)

    xs = xs[indices]
    labels = labels[indices]
    gc.collect()

    # Might as well report a summary of what is in the labels...
    label_summary = dict.fromkeys(range(0, 23), 0)
    for label in labels:
        label_summary[label] += 1
    label_summary = {k: v for k, v in label_summary.items() if v > 0}
    print("Label summary: ", label_summary)

    print("preparing sets...")
    print("- start char and index_from")
    if start_char is not None:
        xs_len = len(xs)
        iterations = 0
        while iterations < xs_len:
            xs[iterations] = [start_char] + \
                [w + index_from for w in xs[iterations]]
            iterations += 1
            if iterations % 1_000_000 == 0:
                print("Iterations: ", iterations)
                gc.collect()
    elif index_from:
        xs_len = int(len(xs))
        iterations = 0
        while iterations < xs_len:
            xs[iterations] = [w + index_from for w in xs[iterations]]
            iterations += 1
            if iterations % 1_000_000 == 0:
                print("Iterations: ", iterations)
                gc.collect()

    if maxlen:
        print("- maxlen")
        xs, labels = _remove_long_seq(maxlen, xs, labels)
    if not num_words:
        num_words = max([max(x) for x in xs])

    print("- oov char")
    # by convention, use 2 as OOV word
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
            if iterations % 1_000_000 == 0:
                print("Iterations: ", iterations)
                gc.collect()

    else:
        xs_len = int(len(xs))
        iterations = 0
        while iterations < xs_len:
            xs[iterations] = [w for w in xs[iterations]
                              if skip_top <= w < num_words]
            iterations += 1
            if iterations % 1_000_000 == 0:
                print("Iterations: ", iterations)
                gc.collect()

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
    with open('data/arxiv_word_index.json') as json_data:
        return json.load(json_data)


def load_glove():
    glove = {}
    with open('data/glove.arxmliv.5B.300d.txt') as glove_data:
        for line in glove_data:
            items = line.split()
            key = items[0]
            glove[key] = np.asarray(items[1:], dtype='float32')
    return glove


def build_embedding_layer(with_input=False, maxlen=256, index_from=2, vocab_dim=300, batch_size=50, mask_zero=True):
    print("loading word embeddings...")
    index_dict = load_vocab()
    word_vectors = load_glove()
    # adding 2 to account for 0th index (for masking), as well as 1st for start and 2nd for oov
    # most frequent word ('the') hence has index 3 in the loaded data, so we start loading the embeddings from index 3 as well
    n_symbols = len(index_dict) + index_from + 1
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index+index_from, :] = word_vectors[word]

    if not with_input:
        # define inputs here
        embedding_layer = Embedding(
            mask_zero=mask_zero,
            output_dim=vocab_dim, input_dim=n_symbols, input_length=maxlen, trainable=False, weights=[embedding_weights])
        # if you don't do this, the next step won't work
        return embedding_layer
    else:
        # define inputs here
        input_1 = Input(shape=(maxlen,), dtype='int32')
        embedding_layer = Embedding(
            weights=[embedding_weights],
            mask_zero=mask_zero,
            output_dim=vocab_dim, input_dim=n_symbols, input_length=maxlen, trainable=False)(input_1)
        return (embedding_layer, input_1)
