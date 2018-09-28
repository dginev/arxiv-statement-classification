# -*- coding: utf-8 -*-
""" Check dataset for overlap and uniqueness
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import json
import warnings
import gc

# We can not load the npz file, as we don't fit the new hash in RAM...
root_para_dir = "/var/local/ams-paragraphs"

labels = sorted(["acknowledgement", "algorithm", "assumption", "caption", "case", "condition", "conjecture", "corollary", "definition", "example",
                 "fact", "lemma", "notation", "other", "paragraph", "problem", "proof", "proposition", "question", "remark", "result", "step", "theorem"])
vocab_file = "data/vocab.txt"
vocab_lines = open(vocab_file, "r").readlines()
w_index = {}
for v_index, line in enumerate(vocab_lines):
    # offset by 1 as the array starts with 0
    w_index[line.split()[0]] = v_index + 1

xs_hash = {}
unique_singles = {}
duplicate_singles = {}

total_paragraphs = 0
total_duplicates = 0
distinct_paragraphs = 0

paragraph_size = {}

# Do the regular traversal done for the .npz map, but instead of creating x,y lists for training,
# compute statistics immediately.
for label_idx, label in enumerate(labels):
    print("Processing dir: ", label)
    label_dir = os.path.join(root_para_dir, label)
    for para_file in os.listdir(label_dir):
        if para_file.endswith(".txt"):
            w_val = []
            words = open(os.path.join(label_dir, para_file),
                         "r").read().split()
            total_paragraphs += 1
            len_key = len(words)

            for word in words:
                if word in w_index:
                    w_val.append(w_index[word])

            paragraph_size[len_key] = 1 + paragraph_size.get(len_key, 0)
            x = ','.join(str(item) for item in w_val)
            xs_hash.setdefault(x, []).append(label)

ps_json = json.dumps(paragraph_size)
open("data/ams_paragraph_sizes.json", "w").write(ps_json)


iteration = 0
for xs_key, val_list in xs_hash.items():
    distinct_paragraphs += 1
    if len(val_list) > 1:
        val_set = set(val_list)
        if len(val_list) != len(val_set):  # repeats, we have duplicates
            if len(val_set) == 1:  # in a single class
                key = str(val_list[0])
                duplicate_singles[key] = duplicate_singles.get(key, 0) + 1
            else:  # in multipe classes
                total_duplicates += 1
        else:
            # No repeats, but multiple classes, count to duplicates
            total_duplicates += 1
    else:
        key = str(val_list[0])
        unique_singles[key] = unique_singles.get(key, 0) + 1
    iteration += 1
    if iteration % 1_000_000 == 0:
        gc.collect()

unique_single_total = 0
print("Paragraphs that appear once in distinct single classes:")
for key, val in unique_singles.items():
    unique_single_total += val
    print("class ", key, " :", val)

dup_single_total = 0
print("Repeats in single classes:")
for key, val in duplicate_singles.items():
    dup_single_total += val
    print("class ", key, " :", val)

print("Total paragraphs analyzed: ", total_paragraphs)
print("Distinct paragraphs analyzed: %d (%.2f percent of total)" %
      (distinct_paragraphs, distinct_paragraphs/total_paragraphs*100))

print("Distinct paragraphs appearing once in corpus: %d (%.2f percent of distinct)" %
      (unique_single_total, (unique_single_total/distinct_paragraphs*100)))
print("Distinct paragraphs, appearing more than once, only in the same class: %d (%.2f percent of distinct)" %
      (dup_single_total, (dup_single_total/distinct_paragraphs*100)))
print("Distinct paragraphs, appearing in 2+ classes: %d (%.2f percent of distinct)" %
      (total_duplicates, (total_duplicates/distinct_paragraphs*100)))
