# -*- coding: utf-8 -*-
"""creates an .npz dataset from the llamapun-induced directory structure of an "AMS environment" dataset

Example: python src/ams_to_npz.py /path/to/vocab.txt /path/to/ams-paragraphs /path/to/destination.npz

on arXiv 08.2018, this script completes in ~1 hour and requires ~32 GB of RAM for the current (naive) in-memory setup

Problem: naively loading the .npz result with np.load(path) allocates ~34 GB of RAM as well, and takes 3 minutes.
         You would need a much more careful setup for machines with lesser RAM capacity.
"""

import os
import numpy as np
import zipfile
import io
import sys
import tarfile
import h5py

# Defaults
vocab_file = "data/vocab.txt"
ams_paragraph_model = "/var/local/ams_paragraphs_arxmliv_08_2018.tar"
destination = "/var/local/ams_paragraphs_arxmliv_08_2018.hdf5"
max_words = 480

argcount = len(sys.argv[1:])
if argcount > 0:
    vocab_file = sys.argv[1]
    if argcount > 1:
        ams_paragraph_model = sys.argv[2]
        if argcount > 2:
            destination = sys.argv[3]
            if argcount > 3:
                max_words = int(sys.argv[4])

# v1, v2
# labels = sorted(["acknowledgement", "algorithm", "assumption", "caption", "case", "condition", "conjecture", "corollary", "definition", "example",
#                  "fact", "lemma", "notation", "other", "paragraph", "problem", "proof", "proposition", "question", "remark", "result", "step", "theorem"])
# v3 -- 29 classes whitelisted by llamapun, alphabetically sorted.
#          each label is recorded via its numeric index in this array,
#          so this source file is normative for mapping the final model back into label names.
# v4 -- 49 classes whitelisted by llamapun, alphabetically sorted.
labels = sorted([
    "abstract", "acknowledgement", "affirmation", "answer", "assumption",
    "bound", "case", "claim", "comment", "conclusion",
    "condition", "conjecture", "constraint", "convention", "corollary",
    "criterion", "definition", "demonstration", "discussion", "example",
    "exercise", "expansion", "expectation", "experiment", "explanation",
    "fact", "hint", "introduction", "issue", "keywords",
    "lemma", "method", "notation", "note", "notice",
    "observation", "overview", "principle", "problem", "proof",
    "proposition", "question", "remark", "result", "rule",
    "solution", "step", "summary", "theorem"])

# w_index is an in-memory loaded vocabulary, produced alongside the GloVe embeddings of a corpus
#    which we use to map the plaintext words into their respective autoincremented *vocabulary index*
vocab_lines = open(vocab_file, "r").readlines()
w_index = {}
for v_index, line in enumerate(vocab_lines):
    # offset by 1 as the array starts with 0
    w_index[line.split()[0]] = v_index + 1

# As is customary for python ML data, we use x_ to denote the data, and y_ to denote the labels.
x_train_paragraphs = []
y_train_labels = []
x_test_paragraphs = []
y_test_labels = []

word_length_report = {}
label_lookup = {}
label_paragraph_count = {}
for label_index, label in enumerate(labels):
    label_lookup[label] = label_index
    label_paragraph_count[label] = 0

paragraph_index = 0
train_index = 0
test_index = 0

tar = tarfile.open(ams_paragraph_model, "r")
# Need: Big chunks for quick continuous access, but still be able to fit the process in average RAM
chunk_size = 100_000  # tune this by hand for smaller data.

fp = h5py.File(destination, "w")
x_train = fp.create_dataset("x_train", (chunk_size, max_words), maxshape=(
    None, max_words), chunks=(chunk_size, max_words), dtype="int")
print("x_train chunks: ", x_train.chunks)

y_train = fp.create_dataset("y_train", (chunk_size,), maxshape=(
    None,), chunks=(chunk_size,), dtype="int")
print("y_train chunks: ", y_train.chunks)

x_test = fp.create_dataset("x_test", (chunk_size, max_words), maxshape=(
    None, max_words), chunks=(chunk_size, max_words), dtype="int")
print("x_test chunks: ", x_test.chunks)

y_test = fp.create_dataset("y_test", (chunk_size,), maxshape=(
    None,), chunks=(chunk_size,), dtype="int")
print("y_test chunks: ", y_test.chunks)


# Iterate over the tar file and stream the vocabulary indexes into the .hdf5 target (fp)
while True:
    tarinfo = tar.next()
    if tarinfo is None:
        break

    w_val = []
    label = tarinfo.name.split('/')[0]
    label_int_value = label_lookup[label]

    words = tar.extractfile(tarinfo).read().decode('utf-8').split()
    for word in words:
        if word in w_index:
            # Convert word to numeric index
            w_val.extend([w_index[word]])
        # else:
            # Should we drop words unseen by the vocabulary, or use an unknown word instead?
            # Drop for now, -1 could be an alternative
            # w_val.append(-1)
            # print("unk: ", word)
    # Only record sentences with at least one word
    # (this was added after testing and realizing there are sentences with all words unknown to w_index)
    paragraph_length = len(w_val)
    if paragraph_length > 0:
        paragraph_index += 1
        label_counter = label_paragraph_count[label] + 1
        label_paragraph_count[label] = label_counter

        if paragraph_length in word_length_report:
            word_length_report[paragraph_length] += 1
        else:
            word_length_report[paragraph_length] = 1

        # Cap at max_words
        w_val = np.array(w_val[:max_words])
        # Pad the paragraph upto the expected max_words size, using 0 as the padding value
        npad = max_words-len(w_val)
        if npad > 0:
            w_val = np.pad(w_val, (0, npad), mode='constant')
        if label_counter % 5 == 0:  # 20% go in the test suite
            x_test_paragraphs.append(w_val)
            y_test_labels.append(label_int_value)
            test_index += 1
            if test_index % chunk_size == 0:
                x_test[-chunk_size:] = x_test_paragraphs[:]
                x_test.resize(x_test.shape[0]+chunk_size, axis=0)
                x_test_paragraphs = []
                print("-- writing hdf5 chunk")
                print("   new x_test size: ", x_test.shape)
                y_test[-chunk_size:] = y_test_labels[:]
                y_test.resize(y_test.shape[0]+chunk_size, axis=0)
                y_test_labels = []
                print("   new y_test size: ", y_test.shape)
        else:
            x_train_paragraphs.append(w_val)
            y_train_labels.append(label_int_value)
            # Flush output hdf5 to disk every `chunk_size` paragraphs,
            # and resize to be able to write more entries
            train_index += 1
            if train_index % chunk_size == 0:
                x_train[-chunk_size:] = x_train_paragraphs[:]
                x_train.resize(x_train.shape[0]+chunk_size, axis=0)
                x_train_paragraphs = []
                print("-- writing hdf5 chunk")
                print("   new x_train size: ", x_train.shape)
                y_train[-chunk_size:] = y_train_labels[:]
                y_train.resize(y_train.shape[0]+chunk_size, axis=0)
                y_train_labels = []
                print("   new y_train size: ", y_train.shape)

        # Technical: drop the tar memoization every 10_000 paragraphs/files in the tar, to remain relatively low in RAM use
        if paragraph_index % 10_000 == 0:
            tar.members = []
            # Use the chance to log some progress:
            # TODO: this can be prettier.
            print("at paragraph %d : " % paragraph_index)
            for label in labels:
                print("-- found %d of %s" %
                      (label_paragraph_count[label], label))
            print("---")
# Done!
tar.close()

# Report on the completed run:
for label in labels:
    print("found %d of %s" % (label_paragraph_count[label], label))

print("---")
print("total collected paragraphs: ", paragraph_index)
print("---")
# print(" word length report")
# print(word_length_report)
