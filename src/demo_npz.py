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


path = 'data/full_ams.npz'
destination = "data/demo_ams.npz"

with np.load(path) as f:
    xs, labels = f['x'], f['y']

# A "Zero Rule" classifier with this restriction will have accuracy of 0.077725805
max_per_class = 50_000
print("reducing data to ", max_per_class, " per class...")
selection_counter = {}
xs_reduced = []
labels_reduced = []

for (idx, label) in enumerate(labels):
    if not(label in selection_counter):
        selection_counter[label] = 0
    if selection_counter[label] < max_per_class:
        selection_counter[label] += 1
        xs_reduced.append(xs[idx])
        labels_reduced.append(label)

saveCompressed(destination, x=xs_reduced, y=labels_reduced)
