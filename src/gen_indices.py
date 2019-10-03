'''Generates the word and label indices needed for the biLSTM and word embeddings

Utility script.

Execute from the root project directory as:
`python3 src/gen_indices.py /path/to/vocab.txt`
'''

import json
import sys

vocab_file = "data/vocab.txt"
word_index_destination = "./data/word_index_2019.json"
label_index_destination = "./data/label_index_2019.json"

argcount = len(sys.argv[1:])
if argcount > 0:
    vocab_file = sys.argv[1]
    if argcount > 1:
        word_index_destination = sys.argv[2]
        if argcount > 2:
            label_index_destination = sys.argv[3]

vocab_lines = open(vocab_file, "r").readlines()
w_index = {}
for v_index, line in enumerate(vocab_lines):
    # offset by 1 as the array starts with 0
    w_index[line.split()[0]] = v_index + 1

w_json = json.dumps(w_index, indent=2)
open(word_index_destination, "w").write(w_json)

# TODO: These are also hardcoded in tar_to_hdf5,
# I need to move them out in a contant json asset
# 2019 -- 46 classes whitelisted by llamapun, alphabetically sorted.
labels = sorted([
    "abstract", "acknowledgement", "analysis", "application", "assumption",
    "background", "caption", "case", "claim", "conclusion", "condition",
    "conjecture", "contribution", "corollary", "data", "dataset",
    "definition", "demonstration", "description", "discussion", "example",
    "experiment", "fact", "future work", "implementation", "introduction",
    "lemma", "methods", "model", "motivation", "notation", "observation",
    "preliminaries", "problem", "proof", "property", "proposition",
    "question", "related work", "remark", "result", "simulation", "step",
    "summary", "theorem", "theory", ])

label_index = {}
for l_ind, label in enumerate(labels):
    label_index[label] = l_ind

l_json = json.dumps(label_index, indent=2)
open(label_index_destination, "w").write(l_json)
