'''Generates the word and label indices needed for the biLSTM and word embeddings

Utility script.

Execute from the root project directory as:
`python3 src/gen_indices.py /path/to/vocab.txt`
'''

import json
import sys


vocab_file = "data/vocab.txt"
word_index_destination = "./data/ams_word_index.json"
label_index_destination = "./data/ams_label_index.json"

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

# TODO: These are also hardcoded in ams_tar_to_hdf5,
# I need to move them out in a contant json asset
labels = sorted(["abstract", "acknowledgement", "algorithm", "assumption", "caption",
                 "case", "conclusion", "condition", "conjecture", "corollary",
                 "definition", "discussion", "example", "fact", "introduction",
                 "lemma", "method", "notation", "other", "paragraph",
                 "problem", "proof", "proposition", "question", "relatedwork",
                 "remark", "result", "step", "theorem"])
label_index = {}
for l_ind, label in enumerate(labels):
    label_index[label] = l_ind

l_json = json.dumps(label_index, indent=2)
open(label_index_destination, "w").write(l_json)
