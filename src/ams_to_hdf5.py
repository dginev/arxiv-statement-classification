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
ams_para_model = "/var/local/ams_paragraphs_arxmliv_08_2018.tar"
destination = "/var/local/full_ams_08_2018"
max_words = 480

argcount = len(sys.argv[1:])
if argcount > 0:
    vocab_file = sys.argv[1]
    if argcount > 1:
        ams_para_model = sys.argv[2]
        if argcount > 2:
            destination = sys.argv[3]
            if argcount > 3:
                max_words = int(sys.argv[4])

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
# Big chunks for quick continuous access, but still be able to fit in RAM
chunk_count = 1_000_000  # tune this by hand for smaller data.

fp = h5py.File(destination+".hdf5", "w")
x_dset = fp.create_dataset("x", (chunk_count, max_words), maxshape=(
    None, max_words), chunks=(chunk_count, max_words))
print("x_dset chunks: ", x_dset.chunks)

y_dset = fp.create_dataset("y", (chunk_count,), maxshape=(
    None,), chunks=(chunk_count,))
print("y_dset chunks: ", y_dset.chunks)

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
            w_val.extend([w_index[word]])
        # else:
            # Should we drop or use a fake number? Drop for now
            # w_val.append(-1)
            # print("unk: ", word)
    if len(w_val) > 0:
        w_val = np.array(w_val[:max_words])
        npad = max_words-len(w_val)
        if npad > 0:
            w_val = np.pad(w_val, (0, npad), mode='constant')
        x_paras.append(w_val)
        y_labels.append(label_idx)
        if para_idx % 10_000 == 0:  # reset members every 100 files, to deallocate memory
            tar.members = []
            print("at paragraph %d : " % para_idx)
            for label in labels:
                print("-- found %d of %s" % (label_para_count[label], label))
            print("---")
            if para_idx % chunk_count == 0:
                x_dset.resize(x_dset.shape[0]+chunk_count, axis=0)
                x_dset[-chunk_count:] = x_paras[:]
                x_paras = []
                print("-- writing hdf5 chunk")
                print("   new x size: ", x_dset.shape)
                y_dset.resize(y_dset.shape[0]+chunk_count, axis=0)
                y_dset[-chunk_count:] = y_labels[:]
                y_labels = []
                print("   new y size: ", y_dset.shape)

tar.close()

for label in labels:
    print("found %d of %s" % (label_para_count[label], label))

print("---")
print("total collected paragraphs: ", para_idx)
