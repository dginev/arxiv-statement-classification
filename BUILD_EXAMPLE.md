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

5. Create index assets for the current vocabulary
    ```
    python3 src/gen_indices.py data/vocab.txt 
    ```

    Creates `data/ams_word_index.json` and `data/ams_label_index.json` to make the index info quickly available to preprocessing and training.

6. Add a symbolic link to the hdf5 full data to avoid passing around arguments all the time
    ```
    ln -s /var/local/ams_paragraphs_08_2018.hdf5 data/full_ams.hdf5
    ```

7.  Here's how we can extract a demo dataset of 1,000 paragraphs per class, ready to be eagerly loaded by Keras as an `.npz` file
    ```
    python3 src/sandbox_data_for_keras.py 1000 data/full_ams.hdf5 data/sandbox_ams_1k.npz
    ```

    Let's also extract a more meaty dataset of 1,000,000 paragraphs per class (on which the live model was trained).
    As this is done with eager loading all data in RAM before serializing (i.e. very naively), you will need a very significant amount of available run 
    to extract 1 million examples per class. In my case it's a little over 40 GB (forty) of RAM.

    ```
    python3 src/sandbox_data_for_keras.py 1000000 data/full_ams.hdf5 data/sandbox_ams_1m.npz
    ```

    Result:
    ```
    checked 11997921 : selected 7320000 paragraphs.
    Final counts:
    Seen:  13000000
    Selected:  7320668
    By label: 
    {18: 1000000, 10: 707396, 15: 1000000, 9: 444300, 28: 1000000, 25: 685064, 21: 1000000, 22: 940455, 12: 257868, 1: 44997, 17: 53698, 14: 15081, 20: 30843, 13: 19374, 8: 46157, 2: 7544, 3: 29661, 6: 1973, 26: 4044, 27: 7818, 23: 8477, 0: 1558, 7: 4054, 11: 896, 24: 1851, 4: 1353, 5: 3664, 16: 372, 19: 2170}
    ---
    saving demo data at data/sandbox_ams_1m.npz ...
    ```

    An unfortunate reality is that since the rarest classes only have less than 1,000 entries (method being the  rarest at 372), even the small demo target has to iterate through the entire hdf5 file in hope of finding more data entries for the rare label classes. Hence even the demo target requires 30 min of runtime, while almost no RAM is used.

8. We can now compute a baseline model using a simple Dense keras Multilayer Perceptron (MLP).

    Warning: Loading the data will eagerly allocate ~60 GB of RAM for the 1m per class case. Additionally, it may take 20-30 minutes of
    data setup before the neural network training is reached, so *always* ensure you test on a small baseline case first.

    ```
    CUDA_VISIBLE_DEVICES=0 python3 src/ams_dense_baseline.py
    ``` 

    At time of writing, defaults to a 1m word dictionary, with 900k max paragraphs per class, as in the main model setup.
    Creates `model-mlp-baseline-big.h5`, 

    Log example (note: the max-per-class is taken from the full 28 classes, before they are mapped down to the confusion classes):
    ```
    -- loading data...
    -- reducing data to  900000  per class...
    -- reducing to 8 label classes
    -- 1000000 iterations
    -- 2000000 iterations
    -- 3000000 iterations
    -- 4000000 iterations
    -- 5000000 iterations
    -- 6000000 iterations
    -- assigning to arrays
    -- Label summary:  {0: 44997, 1: 911482, 2: 3209831, 3: 707396, 4: 257868, 5: 15081, 6: 30843, 7: 1851}
    -- preparing sets...
    -- index_from
    -- 1000000 iterations
    -- 2000000 iterations
    -- 3000000 iterations
    -- 4000000 iterations
    -- 5000000 iterations
    -- performing train/test cutoff
    4143479 train sequences
    1035870 test sequences
    Pad sequences (samples x time)
    x_train shape: (4143479, 480)
    x_test shape: (1035870, 480)
    y_train shape: (4143479,)
    y_test shape: (1035870,)
    -- loading word embeddings, this may take a little while...
    -- known dictionary items:  1000298
    -- embeddings 
    -- setting up model layout...
    -- training model...
    Train on 3314783 samples, validate on 828696 samples
    Epoch 1/10
    3314783/3314783 [==============================] - 993s 300us/step - loss: 0.5884 - sparse_categorical_accuracy: 0.7961 - val_loss: 0.5071 - val_sparse_categorical_accuracy: 0.8196
    Epoch 2/10
    3314783/3314783 [==============================] - 993s 300us/step - loss: 0.5273 - sparse_categorical_accuracy: 0.8176 - val_loss: 0.4666 - val_sparse_categorical_accuracy: 0.8388
    Epoch 3/10
    3314783/3314783 [==============================] - 994s 300us/step - loss: 0.5082 - sparse_categorical_accuracy: 0.8248 - val_loss: 0.4580 - val_sparse_categorical_accuracy: 0.8412
    Epoch 4/10
    3314783/3314783 [==============================] - 994s 300us/step - loss: 0.4962 - sparse_categorical_accuracy: 0.8294 - val_loss: 0.4617 - val_sparse_categorical_accuracy: 0.8371
    Epoch 5/10
    
    ```