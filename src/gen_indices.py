'''Generates the word and label indices needed for the biLSTM and word embeddings

Utility script.

Execute from the root project directory as:
`python3 src/gen_indices.py`
'''

import json

vocab_file = "data/vocab.arxmliv.txt"
vocab_lines = open(vocab_file, "r").readlines()
w_index = {}
for v_index, line in enumerate(vocab_lines):
    # offset by 1 as the array starts with 0
    w_index[line.split()[0]] = v_index + 1

w_json = json.dumps(w_index)
open("data/arxiv_word_index.json", "w").write(w_json)

labels = sorted(["acknowledgement", "algorithm", "assumption", "caption", "case", "condition", "conjecture", "corollary", "definition", "example",
                 "fact", "lemma", "notation", "other", "paragraph", "problem", "proof", "proposition", "question", "remark", "result", "step", "theorem"])
label_index = {}
for l_ind, label in enumerate(labels):
    label_index[label] = l_ind

l_json = json.dumps(label_index)
open("data/ams_label_index.json", "w").write(l_json)
