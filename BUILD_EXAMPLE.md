## Log from my personal build on a fresh checkout (in case instructions are unclear/incomplete)

1. First, prepare the paragraph dataset via [llamapun](https://github.com/KWARC/llamapun):
```
 cd /path/to/llamapun;
 cargo run --release --example corpus_ams_para_model /data/datasets/dataset-arXMLiv-08-2018/ /var/local/ams_paragraphs_arxmliv_08_2018.tar
```

Which results in a `26 GB` tar file.

2. Clone a fresh repository:
```
cd $HOME
git clone git@github.com:dginev/arxiv-ams-env-bilstm.git 
cd arxiv-ams-env-bilstm
```

3. Set up symbolic links to dataset resources:
```
ln -s /data/datasets/embeddings-arXMLiv-08-2018/glove.arxmliv.11B.300d.txt data/glove.model.txt 
ln -s /data/datasets/embeddings-arXMLiv-08-2018/vocab.arxmliv.txt data/vocab.txt
```

4. Extract a vector representation of the paragraphs, and store it in hdf5
```
python3 src/ams_tar_to_hdf5.py data/vocab.txt /var/local/ams_paragraphs_arxmliv_08_2018.tar /var/local/ams_paragraphs_08_2018.hdf5
```

  Runs in ~4GB of RAM, for ~25 minutes. Creates `ams_paragraphs_08_2018.hdf5` which is `44GB` in size.
  Example result report:

```
x_dset chunks:  (1000000, 480)
y_dset chunks:  (1000000,)

[skip]

found 1563 of abstract
found 45111 of acknowledgement
found 7570 of algorithm
found 29754 of assumption
found 1364 of caption
found 3669 of case
found 1984 of conclusion
found 4063 of condition
found 46227 of conjecture
found 445190 of corollary
found 709297 of definition
found 899 of discussion
found 258619 of example
found 19461 of fact
found 15105 of introduction
found 1350134 of lemma
found 374 of method
found 53824 of notation
found 3809424 of other
found 2170 of paragraph
found 30959 of problem
found 2220427 of proof
found 942429 of proposition
found 8489 of question
found 1857 of relatedwork
found 686715 of remark
found 4049 of result
found 7862 of step
found 1322007 of theorem
---
total collected paragraphs:  12029317
```
