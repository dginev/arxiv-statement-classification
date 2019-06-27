# -*- coding: utf-8 -*-
"""extract a demo dataset for ams classification
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
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


max_per_class = 1_000_000
source = "data/full_ams.hdf5"
destination = "data/sandbox_ams_1m.npz"

argcount = len(sys.argv[1:])
if argcount > 0:
    max_per_class = int(sys.argv[1])
    if argcount > 1:
        source = sys.argv[2]
        if argcount > 2:
            destination = sys.argv[3]

print("reducing data to ", max_per_class, " per class...")
total_count = 0
selected_count = 0
selection_counter = {}
xs_reduced = []
labels_reduced = []

datafile = h5py.File(source, 'r')
fx = datafile["x_train"]
fy = datafile["y_train"]

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
print("saving sandbox data at %s ..." % destination)
saveCompressed(destination, x=xs_reduced, y=labels_reduced)


# -- scratch --
# Notes from running this script:

# A "Zero Rule" classifier with this restriction will have accuracy of 0.077725805

# max 2 million does not fit memory, but is of shape:
# Selected:  10734819
#  By label:
#  {13: 2000000, 8: 820683, 11: 1441416, 7: 472174, 22: 1498498, 16: 2000000, 17: 1040920, 19: 794592, 12: 62210, 9: 411880, 6: 48664, 1: 13435, 0: 5063, 10: 21417, 15: 33146, 2: 30595, 20: 3183, 21: 12934, 18: 8582, 5: 4310, 3: 1376, 4: 4998, 14: 4743}

# max 1.5m also does not fit memory, shape:
# Selected:  9734819
#  By label:
#  {13: 1500000, 8: 820683, 11: 1441416, 7: 472174, 22: 1498498, 16: 1500000, 17: 1040920, 19: 794592, 12: 62210, 9: 411880, 6: 48664, 1: 13435, 0: 5063, 10: 21417, 15: 33146, 2: 30595, 20: 3183, 21: 12934, 18: 8582, 5: 4310, 3: 1376, 4: 4998, 14: 4743}

# max 1m also fits! shape:
# Selected:  7753985
#  By label:
#  {13: 1000000, 8: 820683, 11: 1000000, 7: 472174, 22: 1000000, 16: 1000000, 17: 1000000, 19: 794592, 12: 62210, 9: 411880, 6: 48664, 1: 13435, 0: 5063, 10: 21417, 15: 33146, 2: 30595, 20: 3183, 21: 12934, 18: 8582, 5: 4310, 3: 1376, 4: 4998, 14: 4743}

# max 1m on v2 data (first paragraph only selected for envs):
# Selected:  7190074
# By label:
#  {13: 1000000, 8: 693130, 11: 1000000, 7: 437185, 22: 1000000, 16: 1000000, 17: 924838, 19: 668034, 12: 52760, 9: 248706, 6: 45409, 1: 7398, 0: 4786, 10: 19056, 15: 30089, 2: 28879, 20: 2820, 21: 7673, 18: 8337, 5: 3989, 3: 1352, 4: 3607, 14: 2026}

# max 1m on v3 data (first paragraph only selected for envs):
# Seen:  13000000
# Selected:  7320707
# By label:
# {18: 1000000, 10: 707413, 15: 1000000, 9: 444298, 28: 1000000, 21: 1000000, 22: 940472, 25: 685091, 17: 53697, 12: 257863, 8: 46154, 14: 15081, 1: 44998, 2: 7542, 13: 19374, 20: 30840, 3: 29654, 6: 1973, 26: 4044, 27: 7818, 23: 8477, 0: 1558, 7: 4054, 11: 896, 24: 1851, 4: 1353, 5: 3664, 16: 372, 19: 2170}
