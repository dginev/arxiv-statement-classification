## Log from my personal build on a fresh checkout (in case instructions are unclear/incomplete)

1. First, prepare the paragraph dataset via [llamapun](https://github.com/KWARC/llamapun):
    ```
    cd /path/to/llamapun;
    cargo run --release --example corpus_ams_para_model /data/datasets/dataset-arXMLiv-08-2018/ /var/local/ams_paragraphs_arxmliv_08_2018.tar
    ```

    Which results in a `22 GB` tar file.

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

4. Extract a numeric representation of the paragraphs, and store it in hdf5

    ```
    python3 src/ams_tar_to_hdf5.py data/vocab.txt /var/local/ams_paragraphs_arxmliv_08_2018.tar /var/local/ams_paragraphs_08_2018.hdf5
    ```
      Runs in ~1GB of RAM, for ~25 minutes. Creates `ams_paragraphs_08_2018.hdf5` which is `38GB` in size.
      The resulting data resides in 4 separate datasets inside the HDF5 file: `x_train`, `y_train`, `x_test` and `y_test` (split as 80/20 on each label class)

      Example result report:
    ```
    found 1030771 of abstract
    found 162229 of acknowledgement
    found 36 of affirmation
    found 40 of answer
    found 29576 of assumption
    found 47 of bound
    found 3256 of case
    found 89736 of claim
    found 325 of comment
    found 284584 of conclusion
    found 3949 of condition
    found 44892 of conjecture
    found 752 of constraint
    found 2175 of convention
    found 436765 of corollary
    found 236 of criterion
    found 686714 of definition
    found 23043 of demonstration
    found 116650 of discussion
    found 295149 of example
    found 404 of exercise
    found 5 of expansion
    found 13 of expectation
    found 154 of experiment
    found 16 of explanation
    found 17737 of fact
    found 9 of hint
    found 688530 of introduction
    found 41 of issue
    found 1565 of keywords
    found 1320642 of lemma
    found 50969 of method
    found 16610 of notation
    found 4462 of note
    found 4 of notice
    found 18776 of observation
    found 11279 of overview
    found 236 of principle
    found 30367 of problem
    found 2125747 of proof
    found 829066 of proposition
    found 27240 of question
    found 639037 of remark
    found 239929 of result
    found 774 of rule
    found 163 of solution
    found 6910 of step
    found 117 of summary
    found 1287644 of theorem
    ---
    total collected paragraphs:  10529371
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


7. Pre-analysis.
   7.0. We will need a jupyer notebook for the following experiments
    ```
    CUDA_VISIBLE_DEVICES=0 jupyter notebook
    ```

   7.1. `Pre-analysis BiLSTM Confusion Matrix.ipynb`: Run a "confusion matrix analysis" BiLSTM test, to obtain an empirical overview of the data. Some classes are potentially conceptually and linguistically near, (e.g. "observation" and "discussion"),
   and we do not have apriori understanding of the separability of what one would consider "highly specialized" classes. Meanwhile, the task aims to model highly separable classes, so that e.g. "acknowledgement" and "definition" are never confused,
   as well as "proof" vs "conjecture", and so on. As this is the first time this task has been presented, we do our best to obtain a reasonable setup.


8. We can now compute a baseline model using a simple Dense keras Multilayer Perceptron (MLP).
   We will use data generators and work with the full 10.5 million paragraphs from the start, to get as close to possible to optimal baseline performance.

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

    For a BiLSTM example using a cap of 1 million paragraphs for each of the final 8 classes, see [BiLSTM for Paragraph Classification.ipynb](https://github.com/dginev/arxiv-ams-paragraph-classification/blob/master/BiLSTM%20for%20Paragraph%20Classification.ipynb), which can be executed with all prerequisites met at this point of the example.