'''Generates the word and label indices needed for the biLSTM and word embeddings

Utility script.

Execute from the root project directory as:
`python3 src/gen_indices.py /path/to/vocab.txt`
'''

import json
import sys


vocab_file = "/data/datasets/embeddings-arXMLiv-08-2018/vocab.arxmliv.txt"
word_index_destination = "./data/arxiv_word_index.json"
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

w_json = json.dumps(w_index)
open(word_index_destination, "w").write(w_json)

labels = sorted(["acknowledgement", "algorithm", "assumption", "caption", "case", "condition", "conjecture", "corollary", "definition", "example",
                 "fact", "lemma", "notation", "other", "paragraph", "problem", "proof", "proposition", "question", "remark", "result", "step", "theorem"])
label_index = {}
for l_ind, label in enumerate(labels):
    label_index[label] = l_ind

l_json = json.dumps(label_index)
open(label_index_destination, "w").write(l_json)
