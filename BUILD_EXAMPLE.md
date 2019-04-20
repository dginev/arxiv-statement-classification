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
    3314783/3314783 [==============================] - 1055s 318us/step - loss: 0.4878 - sparse_categorical_accuracy: 0.8327 - val_loss: 0.4479 - val_sparse_categorical_accuracy: 0.8461
    Epoch 6/10
    3314783/3314783 [==============================] - 994s 300us/step - loss: 0.4798 - sparse_categorical_accuracy: 0.8358 - val_loss: 0.4604 - val_sparse_categorical_accuracy: 0.8423
    Epoch 7/10
    3314783/3314783 [==============================] - 993s 300us/step - loss: 0.4733 - sparse_categorical_accuracy: 0.8385 - val_loss: 0.4537 - val_sparse_categorical_accuracy: 0.8408
    Epoch 8/10
    3314783/3314783 [==============================] - 992s 299us/step - loss: 0.4678 - sparse_categorical_accuracy: 0.8403 - val_loss: 0.4481 - val_sparse_categorical_accuracy: 0.8454
    Epoch 9/10
    3314783/3314783 [==============================] - 992s 299us/step - loss: 0.4623 - sparse_categorical_accuracy: 0.8426 - val_loss: 0.4428 - val_sparse_categorical_accuracy: 0.8520
    Epoch 10/10
    3314783/3314783 [==============================] - 992s 299us/step - loss: 0.4573 - sparse_categorical_accuracy: 0.8446 - val_loss: 0.4393 - val_sparse_categorical_accuracy: 0.8504
    Model summary:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 480, 300)          300089400 
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 480, 300)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 144000)            0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 480)               69120480  
    _________________________________________________________________
    dense_2 (Dense)              (None, 480)               230880    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 480)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 8)                 3848      
    =================================================================
    Total params: 369,444,608
    Trainable params: 69,355,208
    Non-trainable params: 300,089,400
    _________________________________________________________________
    None
    -- evaluating model on test data...
    sparse_categorical_accuracy: 85.01%
    -- saving model to disk...
    Per-class test measures:
    1035870/1035870 [==============================] - 72s 70us/step
                precision    recall  f1-score   support

            0       0.99      1.00      0.99      8999
            1       0.86      0.82      0.84    182297
            2       0.85      0.96      0.90    641966
            3       0.84      0.56      0.67    141479
            4       0.72      0.45      0.55     51574
            5       0.89      0.48      0.62      3016
            6       0.86      0.39      0.53      6169
            7       0.64      0.65      0.65       370

    micro avg       0.85      0.85      0.85   1035870
    macro avg       0.83      0.66      0.72   1035870
    weighted avg       0.85      0.85      0.84   1035870
    ```