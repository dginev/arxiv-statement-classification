**WARNING:** This is an unpublished experiment, and all claims and data are tentative until noted otherwise.

**NO, Really:** The code is essentially a "scratchpad" of scripts, doing a variety of preprocessing and model training. Expect it to bite, until refactored.

For a high level overview, see [Experimental design](EXPERIMENT.md).

# Additional Setup
Some files exceed GitHub's file size limit and need to be setup by hand (TODO: add utility setup script)
```
	data/full_ams.hdf5
	data/demo_ams.npz
	data/glove.txt
	data/vocab.txt
	models/
```

# Build Steps
See [BUILD_EXAMPLE](BUILD_EXAMPLE.md) for a full personal log on a fresh repo checkout.

```
  ln -s /data/datasets/embeddings-arXMLiv-08-2018/vocab.arxmliv.txt data/vocab.txt
  ln -s /data/datasets/embeddings-arXMLiv-08-2018/glove.arxmliv.11B.300d.txt data/glove.model.txt

  python3 src/ams_tar_to_hdf5.py data/vocab.txt /path/to/ams-paragraphs.tar /path/to/ams-paragraphs.hdf5
  python3 src/gen_indices.py /path/to/vocab.txt
  ln -s /path/to/ams-paragraphs.hdf5 data/full_ams.hdf5
		
  python3 src/sandbox_data_for_keras.py 1000 data/full_ams.hdf5 data/sandbox_ams_1k.npz
  CUDA_VISIBLE_DEVICES=0 python3 src/ams_dense_baseline.py 

  # optional, good for information on paragraph sizes, to estimate a reasonable "maxlen" word count per paragraphs
  python3 src/dataset_check.py

  CUDA_VISIBLE_DEVICES=0 python3 src/ams_bidirectional_lstm.py 
```

# Improvement needs:
 * Remove non-English paragraphs from dataset
