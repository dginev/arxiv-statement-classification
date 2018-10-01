# -*- coding: utf-8 -*-
"""extract a demo dataset for ams classification (50,000 max per class)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import warnings
import os
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


path_x = 'data/full_ams_x.npy'
path_y = 'data/full_ams_y.npy'
destination = "data/demo_ams.npz"

fx = np.memmap(path_x, mode='r')
fy = np.memmap(path_y, mode='r')

# A "Zero Rule" classifier with this restriction will have accuracy of 0.077725805
max_per_class = 50_000
print("reducing data to ", max_per_class, " per class...")
total_paragraphs = 0
selection_counter = {}
xs_reduced = []
labels_reduced = []

for (xs, label) in zip(fx, fy):
    if not(label in selection_counter):
        selection_counter[label] = 0
    if selection_counter[label] < max_per_class:
        selection_counter[label] += 1
        xs_reduced.append(xs)
        labels_reduced.append(label)
        total_paragraphs += 1
        if total_paragraphs % 10_000 == 0:
            print("recorded %d paragraphs..." % total_paragraphs)

print("saving demo data at %s ...", destination)
saveCompressed(destination, x=xs_reduced, y=labels_reduced)
