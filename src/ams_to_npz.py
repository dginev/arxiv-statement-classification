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


def saveCompressed(fh, **namedict):
    with zipfile.ZipFile(fh, mode="w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for k, v in namedict.items():
            with zf.open(k + '.npy', 'w', force_zip64=True) as buf:
                np.lib.npyio.format.write_array(buf,
                                                np.asanyarray(v),
                                                allow_pickle=True)


# Defaults
vocab_file = "data/vocab.txt"
ams_para_model = "/var/local/ams_paragraphs_arxmliv_08_2018.tar"
destination = "/var/local/full_ams_08_2018.npz"


argcount = len(sys.argv[1:])
if argcount > 0:
    vocab_file = sys.argv[1]
    if argcount > 1:
        ams_para_model = sys.argv[2]
        if argcount > 2:
            destination = sys.argv[3]

labels = sorted(["acknowledgement", "algorithm", "assumption", "caption", "case", "condition", "conjecture", "corollary", "definition", "example",
                 "fact", "lemma", "notation", "other", "paragraph", "problem", "proof", "proposition", "question", "remark", "result", "step", "theorem"])
vocab_lines = open(vocab_file, "r").readlines()
w_index = {}
for v_index, line in enumerate(vocab_lines):
    # offset by 1 as the array starts with 0
    w_index[line.split()[0]] = v_index + 1

x_paras = []
y_labels = []

label_lookup = {}
label_para_count = {}
for label_idx, label in enumerate(labels):
    label_lookup[label] = label_idx
    label_para_count[label] = 0

para_idx = 0

tar = tarfile.open(ams_para_model, "r")
while True:
    tarinfo = tar.next()
    if tarinfo is None:
        break
    para_idx += 1

    w_val = []
    label = tarinfo.name.split('/')[0]
    label_idx = label_lookup[label]
    label_para_count[label] += 1

    words = tar.extractfile(tarinfo).read().decode('utf-8').split()
    for word in words:
        if word in w_index:
            w_val.append(w_index[word])
        # else:
            # Should we drop or use a fake number? Drop for now
            # w_val.append(-1)
            # print("unk: ", word)
    x_paras.append(w_val)
    y_labels.append(label_idx)
    if para_idx % 10000 == 0:  # reset members every 100 files, to deallocate memory
        tar.members = []
        print("at paragraph %d : ", para_idx)
        for label in labels:
            print("-- found %d of %s" % (label_para_count[label], label))
        print("---")

tar.close()

for label in labels:
    print("found %d of %s" % (label_para_count[label], label))

saveCompressed(destination, x=x_paras, y=y_labels)
