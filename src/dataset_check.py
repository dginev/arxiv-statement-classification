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
import tarfile

# We can not load the npz file, as we don't fit the new hash in RAM...
ams_para_model = "/var/local/ams_paragraphs_arxmliv_08_2018.tar"

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
label_para_size = {}
for label in labels:
    label_para_size[label] = {}

# Do the regular traversal done for the .npz map, but instead of creating x,y lists for training,
# compute statistics immediately.

tar = tarfile.open(ams_para_model, "r")
while True:
    tarinfo = tar.next()
    if tarinfo is None:
        break
    total_paragraphs += 1
    label = tarinfo.name.split('/')[0]

    words = tar.extractfile(tarinfo).read().decode('utf-8').split()

    w_val = []
    for word in words:
        if word in w_index:
            w_val.append(w_index[word])
    len_key = len(w_val)

    paragraph_size[len_key] = 1 + paragraph_size.get(len_key, 0)
    label_para_size[label][len_key] = 1 + \
        label_para_size[label].get(len_key, 0)

    x = ','.join(str(item) for item in w_val)
    xs_hash.setdefault(x, []).append(label)
    if total_paragraphs % 10_000 == 0:
        print("Processed %d paragraphs." % total_paragraphs)

ps_json = json.dumps(paragraph_size)
open("data/ams_paragraph_sizes.json", "w").write(ps_json)

ls_json = json.dumps(label_para_size)
open("data/ams_label_paragraph_sizes.json", "w").write(ls_json)

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

print("Total paragraphs analyzed: ", total_paragraphs)

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

print("Distinct paragraphs analyzed: %d (%.2f percent of total)" %
      (distinct_paragraphs, distinct_paragraphs/total_paragraphs*100))

print("Distinct paragraphs appearing once in corpus: %d (%.2f percent of distinct)" %
      (unique_single_total, (unique_single_total/distinct_paragraphs*100)))
print("Distinct paragraphs, appearing more than once, only in the same class: %d (%.2f percent of distinct)" %
      (dup_single_total, (dup_single_total/distinct_paragraphs*100)))
print("Distinct paragraphs, appearing in 2+ classes: %d (%.2f percent of distinct)" %
      (total_duplicates, (total_duplicates/distinct_paragraphs*100)))

# arXMLiv 08.2018 results
#
# Total paragraphs analyzed:  35885000
#
# Paragraphs that appear once in distinct single classes:
#
# class  other  : 22169457
# class  definition  : 795301
# class  lemma  : 1420394
# class  corollary  : 465518
# class  theorem  : 1452875
# class  proof  : 5310115
# class  proposition  : 1023122
# class  remark  : 800879
# class  notation  : 60597
# class  example  : 409038
# class  conjecture  : 47282
# class  algorithm  : 12176
# class  acknowledgement  : 4857
# class  fact  : 20566
# class  problem  : 33175
# class  assumption  : 30065
# class  result  : 3072
# class  step  : 12676
# class  question  : 8577
# class  condition  : 4159
# class  caption  : 1002
# class  case  : 4211
# class  paragraph  : 4816
#
# Repeats in single classes:
#
# class  other  : 433783
# class  lemma  : 21593
# class  theorem  : 29770
# class  remark  : 5804
# class  notation  : 1151
# class  proposition  : 15122
# class  proof  : 45053
# class  corollary  : 5673
# class  definition  : 20169
# class  example  : 4557
# class  conjecture  : 940
# class  fact  : 483
# class  acknowledgement  : 115
# class  assumption  : 622
# class  result  : 45
# class  problem  : 398
# class  case  : 245
# class  algorithm  : 450
# class  question  : 87
# class  step  : 207
# class  condition  : 96
# class  paragraph  : 53
# class  caption  : 121
#
# Distinct paragraphs analyzed: 34712798 (96.73 percent of total)
# Distinct paragraphs appearing once in corpus: 34093930 (98.22 percent of distinct)
# Distinct paragraphs, appearing more than once, only in the same class: 586537 (1.69 percent of distinct)
# Distinct paragraphs, appearing in 2+ classes: 32331 (0.09 percent of distinct)
