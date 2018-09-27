# arxiv-ams-env-bilstm
A bidirectional LSTM paragraph classifier for AMS environments

# Additional Setup
Some files exceed GitHub's file size limit and need to be setup by hand (TODO: add utility setup script)
```
	data/full_ams.npz
	data/demo_ams.npz
	data/glove.txt
	data/vocab.txt
	models/
```

# Build Steps
```
	python3 src/ams_to_npz.py /path/to/vocab.txt /path/to/ams-paragraphs /path/to/destination.npz
	python3 src/gen_indices.py /path/to/vocab.txt

```