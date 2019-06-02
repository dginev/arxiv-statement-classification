# -*- coding: utf-8 -*-
""" Reduces the 49 class dataset from arXiv down to the confusion-motivated 12 classes (based on 24 of the original labels)

Use: python3 src/gen_confusion_free_dataset.py

"""

import warnings
import numpy as np
import gc
import json
import h5py
import time
import sys


# We run `Pre-analysis BiLSTM Confusion Matrix.ipynb`

# Confusion matrix obtained during pre-analysis
original_class_names = [
    'abstract', 'acknowledgement', 'affirmation', 'answer', 'assumption', 'bound',
    'case', 'claim', 'comment', 'conclusion', 'condition', 'conjecture',
    'constraint', 'convention', 'corollary', 'criterion', 'definition',
    'demonstration', 'discussion', 'example', 'exercise', 'expansion',
    'expectation', 'experiment', 'explanation', 'fact', 'hint', 'introduction',
    'issue', 'keywords', 'lemma', 'method', 'notation', 'note', 'notice',
    'observation', 'overview', 'principle', 'problem', 'proof', 'proposition',
    'question', 'relatedwork', 'remark', 'result', 'rule', 'solution', 'step', 'summary', 'theorem']

reduced_class_names = [
    'abstract', 'acknowledgement', 'conclusion', 'definition', 'example',
    'introduction', 'keywords', 'proof', 'proposition', 'problem', 'relatedwork', 'remark', 'result']

confusion_map_names = {
    "abstract": "abstract",
    "acknowledgement": "acknowledgement",
    "conclusion": "conclusion",
    "discussion": "conclusion",
    "definition": "definition",
    "example": "example",
    "introduction": "introduction",
    "keywords": "keywords",
    "proof": "proof",
    "demonstration": "proof",
    "lemma": "proposition",
    "theorem": "proposition",
    "proposition": "proposition",
    "assumption": "proposition",
    "condition": "proposition",
    "fact": "proposition",
    "conjecture": "proposition",
    "claim": "proposition",
    "corollary": "proposition",
    "question": "problem",
    "problem": "problem",
    "relatedwork": "relatedwork",
    "remark": "remark",
    "note": "remark",
    "result": "result"
}

confusion_map = {}
for k, v in confusion_map_names.items():
    confusion_map[original_class_names.index(k)] = reduced_class_names.index(v)

# Ok, now that we have the confusion map. we need to remap and rewrite the HDF5 data.
# We do this once here so that we can quickly train models afterwards, rather than having to recompute the map each time
input_filename = "data/full_ams.hdf5"
output_filename = "data/confusion_free_ams.hdf5"

argcount = len(sys.argv[1:])
if argcount > 0:
    input_filename = sys.argv[1]
    if argcount > 1:
        output_filename = sys.argv[2]

data_hf = h5py.File(input_filename, 'r')

# Need: Big chunks for quick continuous access, but still be able to fit the process in average RAM
chunk_size = 100_000  # tune this by hand for smaller data.
max_words = 480  # explicit, to fail loudly if something changes
new_fp = h5py.File(output_filename, "w")


total_train_size = 0
total_test_size = 0

print("Original train size: %d " % data_hf["y_train"].shape[0])
index = 0
for label in data_hf["y_train"]:
    index += 1
    if label in confusion_map:
        total_train_size += 1
        if total_train_size % 100_000 == 0:
            print("Collected %d train entries from %d data entries." %
                  (total_train_size, index))

print("New train size: %d " % total_train_size)

print("Original test size: %d " % data_hf["y_test"].shape[0])
index = 0
for label in data_hf["y_test"]:
    index += 1
    if label in confusion_map:
        total_test_size += 1
        if total_test_size % 100_000 == 0:
            print("Collected %d test entries from %d data entries." %
                  (total_test_size, index))

print("New test size: %d " % total_test_size)

x_train = new_fp.create_dataset("x_train", (total_train_size, max_words), maxshape=(
    None, max_words), chunks=(chunk_size, max_words), dtype="int")
print("x_train chunks: ", x_train.chunks)

y_train = new_fp.create_dataset("y_train", (total_train_size,), maxshape=(
    None,), chunks=(chunk_size,), dtype="int")
print("y_train chunks: ", y_train.chunks)

x_test = new_fp.create_dataset("x_test", (total_test_size, max_words), maxshape=(
    None, max_words), chunks=(chunk_size, max_words), dtype="int")
print("x_test chunks: ", x_test.chunks)

y_test = new_fp.create_dataset("y_test", (total_test_size,), maxshape=(
    None,), chunks=(chunk_size,), dtype="int")
print("y_test chunks: ", y_test.chunks)

# Looping twice, but code is simpler...
new_index = 0
for (index, label) in enumerate(data_hf["y_train"]):
    if label in confusion_map:
        new_index += 1
        if new_index >= total_train_size:
            # print("Error: more train items than allocated (at index %d)" % new_index)
            break
        x_train[new_index] = data_hf["x_train"][index]
        y_train[new_index] = confusion_map[label]

new_index = 0
for (index, label) in enumerate(data_hf["y_test"]):
    if label in confusion_map:
        new_index += 1
        if new_index >= total_test_size:
            # print("Error: more test items than allocated (at index %d)" % new_index)
            break
        x_test[new_index] = data_hf["x_test"][index]
        y_test[new_index] = confusion_map[label]

data_hf.close()
new_fp.close()
