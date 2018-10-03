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
import h5py


def saveCompressed(fh, **namedict):
    with zipfile.ZipFile(fh, mode="w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for k, v in namedict.items():
            with zf.open(k + '.npy', 'w', force_zip64=True) as buf:
                np.lib.npyio.format.write_array(buf,
                                                np.asanyarray(v),
                                                allow_pickle=True)


source = "data/full_ams.hdf5"
destination = "data/demo_ams.npz"

# A "Zero Rule" classifier with this restriction will have accuracy of 0.077725805
max_per_class = 250_000
print("reducing data to ", max_per_class, " per class...")
total_count = 0
selected_count = 0
selection_counter = {}
xs_reduced = []
labels_reduced = []

datafile = h5py.File(source, 'r')
fx = datafile["x"]
fy = datafile["y"]

for xs, label in zip(fx, fy):
    total_count += 1
    if xs[0] == 0:
        continue  # skip padded storage
    if not(label in selection_counter):
        selection_counter[label] = 0
    if selection_counter[label] < max_per_class:
        xs_reduced.append(xs)
        labels_reduced.append(label)

        selection_counter[label] += 1
        selected_count += 1
        if selected_count % 10_000 == 0:
            print("checked %d : selected %d paragraphs." %
                  (total_count, selected_count))
            # print("label: ", label)
            # print("para: ", xs)
            # print("---")
    elif all([x >= max_per_class for x in selection_counter.values()]):
        break

print("Final counts:")
print("Seen: ", total_count)
print("Selected: ", selected_count)
print("By label: ")
print(selection_counter)
print("---")
print("saving demo data at %s ..." % destination)
saveCompressed(destination, x=xs_reduced, y=labels_reduced)
