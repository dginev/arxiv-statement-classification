# -*- coding: utf-8 -*-
"""creates an .npz dataset from the llamapun-induced directory structure of an "AMS environment" dataset

on arXiv, this script completes in ~1 hour and requires ~32 GB of RAM for the current (naive) in-memory setup

Problem: naively loading the .npz result with np.load(path) allocates ~34 GB of RAM as well, and takes 3 minutes.
         You would need a much more careful setup for machines with lesser RAM capacity.
"""

import os
import numpy as np
import zipfile
import io


def saveCompressed(fh, **namedict):
    with zipfile.ZipFile(fh, mode="w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for k, v in namedict.items():
            with zf.open(k + '.npy', 'w', force_zip64=True) as buf:
                np.lib.npyio.format.write_array(buf,
                                                np.asanyarray(v),
                                                allow_pickle=True)


root_para_dir = "/var/local/ams-paragraphs"
destination = "arxiv_ams.npz"

labels = sorted(["acknowledgement", "algorithm", "assumption", "caption", "case", "condition", "conjecture", "corollary", "definition", "example",
                 "fact", "lemma", "notation", "other", "paragraph", "problem", "proof", "proposition", "question", "remark", "result", "step", "theorem"])
vocab_file = "../data/vocab.arxmliv.txt"
vocab_lines = open(vocab_file, "r").readlines()
w_index = {}
for v_index, line in enumerate(vocab_lines):
    # offset by 1 as the array starts with 0
    w_index[line.split()[0]] = v_index + 1

x_paras = []
y_labels = []

for label_idx, label in enumerate(labels):
    print("Processing dir: ", label)
    label_dir = os.path.join(root_para_dir, label)
    for para_file in os.listdir(label_dir):
        if para_file.endswith(".txt"):
            w_val = []
            words = open(os.path.join(label_dir, para_file),
                         "r").read().split()
            for word in words:
                if word in w_index:
                    w_val.append(w_index[word])
                # else:
                    # Should we drop or use a fake number? Drop for now
                    # w_val.append(-1)
                    # print("unk: ", word)
            x_paras.append(w_val)
            y_labels.append(label_idx)


saveCompressed(destination, x=x_paras, y=y_labels)
