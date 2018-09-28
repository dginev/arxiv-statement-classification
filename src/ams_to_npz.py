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
vocab_file = "/data/vocab.txt"
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

tar = tarfile.open(ams_para_model, "r")
paragraphs = tar.getmembers()

for label_idx, label in enumerate(labels):
    print("Processing dir: ", label)
    label_paragraphs = [para for para in paragraphs
                        if para.name.startswith(label)]
    print("found %d paragraphs" % len(label_paragraphs))
    for paragraph in label_paragraphs:
        w_val = []
        words = tar.extractfile(paragraph).read().decode('utf-8').split()
        for word in words:
            if word in w_index:
                w_val.append(w_index[word])
            # else:
                # Should we drop or use a fake number? Drop for now
                # w_val.append(-1)
                # print("unk: ", word)
        x_paras.append(w_val)
        y_labels.append(label_idx)

tar.close()

saveCompressed(destination, x=x_paras, y=y_labels)
