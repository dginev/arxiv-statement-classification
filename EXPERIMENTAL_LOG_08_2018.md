AMS Paragraphs from arXiv 08.2018

# 23 classes
1. Baseline, Dense(maxlen) x 2, 23 classes - 0.31 f1-score

2. BiLSTM(maxlen/2) x 2, 23 classes - 0.49 f1-score

  7m Paragraphs, 50k cap,
  Label summary:  {0: 1004, 1: 2405, 2: 5968, 3: 302, 4: 712, 5: 761, 6: 9680, 7: 49988, 8: 49980, 9: 49870, 10: 4027, 11: 49977, 12: 11526, 13: 49948, 14: 1110, 15: 6443, 16: 49984, 17: 49975, 18: 1728, 19: 49993, 20: 893, 21: 2376, 22: 49988}
  366906 train sequences
  91727 test sequences
  known dictionary items:  1000298

  ```
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  embedding_1 (Embedding)      (None, 300, 300)          300089400 
  _________________________________________________________________
  dropout_1 (Dropout)          (None, 300, 300)          0         
  _________________________________________________________________
  bidirectional_1 (Bidirection (None, 300, 300)          541200    
  _________________________________________________________________
  bidirectional_2 (Bidirection (None, 300)               541200    
  _________________________________________________________________
  dropout_2 (Dropout)          (None, 300)               0         
  _________________________________________________________________
  dense_1 (Dense)              (None, 23)                6923      
  =================================================================
  Total params: 301,178,723
  Trainable params: 1,089,323
  Non-trainable params: 300,089,400
  ```

  ```
  91727/91727 [==============================] - 1611s 18ms/step
              precision    recall  f1-score   support

            0       0.75      0.98      0.85       191
            1       0.78      0.64      0.70       433
            2       0.50      0.47      0.49      1157
            3       0.92      0.58      0.71        62
            4       0.70      0.29      0.41       148
            5       0.80      0.05      0.09       166
            6       0.37      0.14      0.21      1883
            7       0.34      0.35      0.35      9668
            8       0.70      0.85      0.77      9659
            9       0.65      0.63      0.64      8343
          10       0.55      0.07      0.13       796
          11       0.35      0.52      0.42      9648
          12       0.46      0.36      0.40      2183
          13       0.54      0.51      0.53      9035
          14       0.76      0.31      0.44       236
          15       0.70      0.52      0.60      1291
          16       0.67      0.73      0.70      7687
          17       0.28      0.27      0.27      9464
          18       0.26      0.06      0.10       318
          19       0.61      0.64      0.63      9373
          20       0.63      0.24      0.34       161
          21       0.41      0.09      0.15       420
          22       0.41      0.27      0.33      9405

  avg / total       0.50      0.50      0.49     91727
  ```

# 9 classes

0. 50k data, 7m para source
  Label summary:  {0: 1004, 1: 2405, 2: 302, 3: 49984, 4: 5968, 5: 49980, 6: 6443, 7: 49993, 8: 332559}

1. Baseline, Dense(maxlen) x 2, 9 classes - 0.67 f1-score

2. Fitting fidgeting
 - 5k max class (7m para data)
  - BiLSTM(maxlen/2), f1: 0.76
  - BiLSTM(maxlen/2) x 2, f1:  0.76
  - BiLSTM(maxlen), f1: 0.76
 - 50k max class (7m para data)
  - BiLSTM(maxlen/2) + Dense(maxlen/2) + Dense(maxlen/4), f1: 0.81
  - BiLSTM(maxlen/4), f1: 0.81
  - BiLSTM(maxlen/8), f1:0.80
  - BiLSTM(16), f1: 0.79



3. BiLSTM(maxlen/2) x 2, 5k max class, Weighted classes, stratified test/train sets

 - Same 0.77 f1-score results with a single BiLSTM(maxlen/4) !!!
   ```
      21722/21722 [==============================] - 38s 2ms/step
              precision    recall  f1-score   support

              0       0.99      1.00      0.99      1000
              1       0.86      0.78      0.81      1000
              2       0.93      0.75      0.83       275
              3       0.63      0.41      0.50      1000
              4       0.55      0.40      0.46      1000
              5       0.61      0.33      0.43      1000
              6       0.76      0.40      0.52      1000
              7       0.59      0.29      0.39      1000
              8       0.80      0.92      0.86     14447
      avg/total       0.77      0.79      0.77     21722
    ```

```
    Label summary:  {0: 1004, 1: 2405, 2: 302, 3: 5000, 4: 5000, 5: 5000, 6: 5000, 7: 5000, 8: 51607}

    x_train shape: (62905, 480)
    x_test shape: (15727, 480)
    y_train shape: (62905,)
    y_test shape: (15727,)
    known dictionary items:  1000298

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 480, 300)          300089400 
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 480, 300)          0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 480, 480)          1038720   
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 480, 480)          0         
    _________________________________________________________________
    bidirectional_2 (Bidirection (None, 480)               1384320   
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 480)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 9)                 4329      
    =================================================================
    Total params: 302,516,769
    Trainable params: 2,427,369
    Non-trainable params: 300,089,400
    _________________________________________________________________

    Train on 50324 samples, validate on 12581 samples
    Epoch 1/10
    50324/50324 [==============================] - 1151s 23ms/step - loss: 0.9839 - weighted_sparse_categorical_accuracy: 0.6785 - val_loss: 0.8223 - val_weighted_sparse_categorical_accuracy: 0.6998
    Epoch 2/10
    50324/50324 [==============================] - 1175s 23ms/step - loss: 0.7530 - weighted_sparse_categorical_accuracy: 0.7291 - val_loss: 0.7326 - val_weighted_sparse_categorical_accuracy: 0.7278
    Epoch 3/10
    50324/50324 [==============================] - 1162s 23ms/step - loss: 0.6623 - weighted_sparse_categorical_accuracy: 0.7556 - val_loss: 0.6722 - val_weighted_sparse_categorical_accuracy: 0.7529
    Epoch 4/10
    50324/50324 [==============================] - 1169s 23ms/step - loss: 0.5979 - weighted_sparse_categorical_accuracy: 0.7776 - val_loss: 0.6544 - val_weighted_sparse_categorical_accuracy: 0.7596
    Epoch 5/10
    50324/50324 [==============================] - 1186s 24ms/step - loss: 0.5392 - weighted_sparse_categorical_accuracy: 0.8003 - val_loss: 0.6346 - val_weighted_sparse_categorical_accuracy: 0.7701
    Epoch 6/10
    50324/50324 [==============================] - 1174s 23ms/step - loss: 0.4810 - weighted_sparse_categorical_accuracy: 0.8209 - val_loss: 0.6580 - val_weighted_sparse_categorical_accuracy: 0.7588
    Epoch 7/10
    50324/50324 [==============================] - 1169s 23ms/step - loss: 0.4213 - weighted_sparse_categorical_accuracy: 0.8437 - val_loss: 0.6962 - val_weighted_sparse_categorical_accuracy: 0.7596
    Epoch 8/10
    50324/50324 [==============================] - 1163s 23ms/step - loss: 0.3586 - weighted_sparse_categorical_accuracy: 0.8670 - val_loss: 0.6826 - val_weighted_sparse_categorical_accuracy: 0.7764
    Epoch 9/10
    50324/50324 [==============================] - 1162s 23ms/step - loss: 0.2913 - weighted_sparse_categorical_accuracy: 0.8936 - val_loss: 0.7434 - val_weighted_sparse_categorical_accuracy: 0.7659
    Epoch 10/10
    50324/50324 [==============================] - 1163s 23ms/step - loss: 0.2367 - weighted_sparse_categorical_accuracy: 0.9141 - val_loss: 0.7788 - val_weighted_sparse_categorical_accuracy: 0.7733
    Evaluating model on test data...
    weighted_sparse_categorical_accuracy: 76.99%

    Per-class test measures:
    15727/15727 [==============================] - 438s 28ms/step
                precision    recall  f1-score   support

              0       0.97      0.99      0.98       201
              1       0.81      0.82      0.82       458
              2       0.87      0.67      0.75        60
              3       0.64      0.64      0.64       911
              4       0.60      0.63      0.62       990
              5       0.67      0.49      0.57       993
              6       0.74      0.53      0.62       994
              7       0.51      0.56      0.53       984
              8       0.83      0.86      0.84     10136

    avg / total       0.77      0.77      0.77     15727
```

3.1.  BiLSTM(maxlen/2) x 2, 50k max class, Weighted classes, stratified test/train sets

```
    Label summary:  {0: 1004, 1: 2405, 2: 302, 3: 49984, 4: 5968, 5: 49980, 6: 6443, 7: 49993, 8: 332559}

    x_train shape: (387671, 480)
    x_test shape: (96918, 480)
    y_train shape: (387671,)
    y_test shape: (96918,)

    known dictionary items:  1000298    

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 480, 300)          300089400 
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 480, 300)          0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 480, 480)          1038720   
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 480, 480)          0         
    _________________________________________________________________
    bidirectional_2 (Bidirection (None, 480)               1384320   
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 480)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 9)                 4329      
    =================================================================
    Total params: 302,516,769
    Trainable params: 2,427,369
    Non-trainable params: 300,089,400
    _________________________________________________________________
    None
    Train on 310136 samples, validate on 77535 samples
    Epoch 1/10
    310136/310136 [==============================] - 7214s 23ms/step - loss: 0.6519 - weighted_sparse_categorical_accuracy: 0.7691 - val_loss: 0.5456 - val_weighted_sparse_categorical_accuracy: 0.8033
    Epoch 2/10
    310136/310136 [==============================] - 7089s 23ms/step - loss: 0.5258 - weighted_sparse_categorical_accuracy: 0.8100 - val_loss: 0.5168 - val_weighted_sparse_categorical_accuracy: 0.8153
    Epoch 3/10
    310136/310136 [==============================] - 7011s 23ms/step - loss: 0.4827 - weighted_sparse_categorical_accuracy: 0.8245 - val_loss: 0.4986 - val_weighted_sparse_categorical_accuracy: 0.8202
    Epoch 4/10
    310136/310136 [==============================] - 7008s 23ms/step - loss: 0.4471 - weighted_sparse_categorical_accuracy: 0.8365 - val_loss: 0.4869 - val_weighted_sparse_categorical_accuracy: 0.8249
    Epoch 5/10
    310136/310136 [==============================] - 7046s 23ms/step - loss: 0.4131 - weighted_sparse_categorical_accuracy: 0.8490 - val_loss: 0.4827 - val_weighted_sparse_categorical_accuracy: 0.8274
    Epoch 6/10
    310136/310136 [==============================] - 7120s 23ms/step - loss: 0.3777 - weighted_sparse_categorical_accuracy: 0.8613 - val_loss: 0.4915 - val_weighted_sparse_categorical_accuracy: 0.8261
    Epoch 7/10
    310136/310136 [==============================] - 7115s 23ms/step - loss: 0.3427 - weighted_sparse_categorical_accuracy: 0.8735 - val_loss: 0.4991 - val_weighted_sparse_categorical_accuracy: 0.8276
    Epoch 8/10
    310136/310136 [==============================] - 7033s 23ms/step - loss: 0.3109 - weighted_sparse_categorical_accuracy: 0.8854 - val_loss: 0.5219 - val_weighted_sparse_categorical_accuracy: 0.8241
    Epoch 9/10
    310136/310136 [==============================] - 7080s 23ms/step - loss: 0.2808 - weighted_sparse_categorical_accuracy: 0.8967 - val_loss: 0.5491 - val_weighted_sparse_categorical_accuracy: 0.8238
    Epoch 10/10
    310136/310136 [==============================] - 7154s 23ms/step - loss: 0.2555 - weighted_sparse_categorical_accuracy: 0.9050 - val_loss: 0.5783 - val_weighted_sparse_categorical_accuracy: 0.82199059
    Evaluating model on test data...
    weighted_sparse_categorical_accuracy: 82.26%

    Per-class test measures:
    96918/96918 [==============================] - 2730s 28ms/step
                precision    recall  f1-score   support

              0       0.78      0.94      0.85       201
              1       0.81      0.77      0.79       458
              2       0.83      0.82      0.82        60
              3       0.71      0.71      0.71      8968
              4       0.57      0.41      0.47      1182
              5       0.80      0.76      0.78      9910
              6       0.65      0.63      0.64      1281
              7       0.64      0.59      0.61      9820
              8       0.87      0.89      0.88     65038

    avg / total       0.82      0.82      0.82     96918
```

4. BiLSTM(120) x 2, batch 256, 50k max class, improved demo dataset

```
  Label summary:  {0: 5063, 1: 13435, 2: 1376, 3: 50000, 4: 30595, 5: 50000, 6: 33146, 7: 50000, 8: 458831}


  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  embedding_1 (Embedding)      (None, 480, 300)          300089400 
  _________________________________________________________________
  dropout_1 (Dropout)          (None, 480, 300)          0         
  _________________________________________________________________
  bidirectional_1 (Bidirection (None, 480, 240)          404160    
  _________________________________________________________________
  dropout_2 (Dropout)          (None, 480, 240)          0         
  _________________________________________________________________
  bidirectional_2 (Bidirection (None, 240)               346560    
  _________________________________________________________________
  dropout_3 (Dropout)          (None, 240)               0         
  _________________________________________________________________
  dense_1 (Dense)              (None, 9)                 2169      
  =================================================================
  Total params: 300,842,289
  Trainable params: 752,889
  Non-trainable params: 300,089,400
  _________________________________________________________________
  None
  Train on 443164 samples, validate on 110792 samples


  Epoch 10/10
  443164/443164 [==============================] - 5247s 12ms/step - loss: 0.4497 - weighted_sparse_categorical_accuracy: 0.8317 - val_loss: 0.5464 - val_weighted_sparse_categorical_accuracy: 0.7981

  Epoch 00010: val_weighted_sparse_categorical_accuracy did not improve from 0.79946
  Saving model to disk : bilstm-120-dual-9cat-bigbatch 
  Per-class test measures:
  138490/138490 [==============================] - 496s 4ms/step
              precision    recall  f1-score   support

            0       0.93      0.99      0.96      1013
            1       0.81      0.79      0.80      2687
            2       0.85      0.74      0.79       275
            3       0.70      0.70      0.70     10000
            4       0.62      0.67      0.65      6119
            5       0.68      0.67      0.67     10000
            6       0.77      0.61      0.68      6629
            7       0.63      0.57      0.60     10000
            8       0.86      0.87      0.86     91767

  avg / total       0.80      0.80      0.80    138490
```

5. 5k max, experiments, maxlen: 480, 256 batch

 `score (precision, recall, f1, support)`

 - BiLSTM(120):
    `0.78      0.79      0.78     21722`

 - BiLSTM(120) + Dense(120):
    `0.78      0.79      0.78     21722`
 
 - BiLSTM(120)x2 + Dense(120): 
    `0.78      0.79      0.78     21722`
 
 - BiLSTM(120) + TimeDistributed(Dense(240)) + BiLSTM(120) + Dense(120):
    `0.79      0.79      0.78     21722`

 - BiLSTM(120) + BiLSTM(60) (with varied merge_mode, consistent):
    `0.78      0.79      0.78     21722`
 - BiLSTM(64):
    `0.78      0.79      0.78     21722`
 - BiLSTM(32):
    `0.76      0.78      0.76     21722`
 - BiLSTM(16):
    `0.76      0.78      0.75     21722` 
 - BiLSTM(8):
    `0.76      0.78      0.75     21722` 
 - BiLSTM(1):
    `0.73      0.76      0.71     21722`
 - BiLSTM(128), 8 classes (no assumption, yes remark)
    `0.83      0.84      0.83     21722`

    ```
      Per-class test measures:
      21722/21722 [==============================] - 39s 2ms/step
                  precision    recall  f1-score   support

                0       0.99      1.00      1.00      1000
                1       0.90      0.76      0.82      1000
                2       0.91      0.76      0.83       275
                3       0.68      0.39      0.49      1000
                4       0.57      0.53      0.55      1000
                5       0.70      0.46      0.55      1000
                6       0.53      0.45      0.48      1000
                7       0.86      0.93      0.90     15447
    ```

- BiLSTM(128), 7 classes (no assumption, no remark)
    `0.87      0.87      0.87     21722`

    ```
    Per-class test measures:
    21722/21722 [==============================] - 38s 2ms/step
                precision    recall  f1-score   support

              0       0.98      1.00      0.99      1000
              1       0.88      0.78      0.83      1000
              2       0.92      0.75      0.83       275
              3       0.64      0.47      0.54      1000
              4       0.56      0.53      0.54      1000
              5       0.76      0.39      0.52      1000
              6       0.90      0.95      0.92     16447
    ```

- BiLSTM(128) + BiLSTM(64), 7 classes (no assumption, no remark)
   `0.87      0.87      0.87     21722`

- BiLSTM(128) + Conv1D(100,3) + MaxPooling1D(2) (no assumption, no remark)
  ```
  Epoch 00010: val_weighted_sparse_categorical_accuracy did not improve from 0.86753
  Saving model to disk : bilstm128_2Dcnn_batch256_cat7 
  Per-class test measures:
  21722/21722 [==============================] - 34s 2ms/step
              precision    recall  f1-score   support

            0       0.98      0.99      0.99      1000
            1       0.89      0.76      0.82      1000
            2       0.76      0.82      0.79       275
            3       0.53      0.47      0.50      1000
            4       0.56      0.48      0.52      1000
            5       0.68      0.46      0.55      1000
            6       0.90      0.93      0.91     16447

  avg / total       0.86      0.86      0.86     21722
  ```



6. BiLSTM(120) x 2, batch 256, 250k max class
```
  Label summary:  {0: 5063, 1: 13435, 2: 1376, 3: 250000, 4: 30595, 5: 250000, 6: 33146, 7: 250000, 8: 1671041}

  x_train shape: (2003724, 480)
  x_test shape: (500932, 480)
  y_train shape: (2003724,)
  y_test shape: (500932,)

  known dictionary items:  1000298

  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  embedding_1 (Embedding)      (None, 480, 300)          300089400 
  _________________________________________________________________
  dropout_1 (Dropout)          (None, 480, 300)          0         
  _________________________________________________________________
  bidirectional_1 (Bidirection (None, 480, 240)          404160    
  _________________________________________________________________
  dropout_2 (Dropout)          (None, 480, 240)          0         
  _________________________________________________________________
  bidirectional_2 (Bidirection (None, 240)               346560    
  _________________________________________________________________
  dropout_3 (Dropout)          (None, 240)               0         
  _________________________________________________________________
  dense_1 (Dense)              (None, 9)                 2169      
  =================================================================
  Total params: 300,842,289
  Trainable params: 752,889
  Non-trainable params: 300,089,400
  _________________________________________________________________
  None
  Train on 1602979 samples, validate on 400745 samples
  Epoch 1/10
  1602979/1602979 [==============================] - 18665s 12ms/step - loss: 0.5868 - weighted_sparse_categorical_accuracy: 0.7911 - val_loss: 0.5051 - val_weighted_sparse_categorical_accuracy: 0.8178

  [...]

  Epoch 10/10
  1602979/1602979 [==============================] - 18762s 12ms/step - loss: 0.4243 - weighted_sparse_categorical_accuracy: 0.8456 - val_loss: 0.4547 - val_weighted_sparse_categorical_accuracy: 0.8359

  test evaluation:
    500932/500932 [==============================] - 1770s 4ms/step
              precision    recall  f1-score   support

              0       0.80      0.95      0.87      1013
              1       0.84      0.67      0.74      2687
              2       0.75      0.66      0.70       275
              3       0.74      0.76      0.75     50000
              4       0.64      0.43      0.51      6119
              5       0.79      0.80      0.79     50000
              6       0.80      0.49      0.61      6629
              7       0.69      0.59      0.63     50000
              8       0.88      0.90      0.89    334209

    avg / total       0.83      0.84      0.83    500932
```

7. BiLSTM(128), batch 256, 1m max class

  - zero-rule baseline is 0.76 for class other(6)
  
```
6203188 train sequences
1550797 test sequences
x_train shape: (6203188, 480)
x_test shape: (1550797, 480)
y_train shape: (6203188,)
y_test shape: (1550797,)
loading word embeddings...
known dictionary items:  1000298
setting up model layout...
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 480, 300)          300089400 
_________________________________________________________________
dropout_1 (Dropout)          (None, 480, 300)          0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 256)               439296    
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 7)                 1799      
=================================================================
Total params: 300,530,495
Trainable params: 441,095
Non-trainable params: 300,089,400
_________________________________________________________________
None
Training model...
Train on 4962550 samples, validate on 1240638 samples
Epoch 1/50
4962550/4962550 [==============================] - 30292s 6ms/step - loss: 0.3205 - weighted_sparse_categorical_accuracy: 0.8823 - val_loss: 0.2856 - val_weighted_sparse_categorical_accuracy: 0.8957

Epoch 00001: val_weighted_sparse_categorical_accuracy improved from -inf to 0.89572, saving model to bilstm128_batch256_cat7-checkpoint.h5
Epoch 2/50
4962550/4962550 [==============================] - 30392s 6ms/step - loss: 0.2907 - weighted_sparse_categorical_accuracy: 0.8948 - val_loss: 0.2788 - val_weighted_sparse_categorical_accuracy: 0.8989

Epoch 00002: val_weighted_sparse_categorical_accuracy improved from 0.89572 to 0.89894, saving model to bilstm128_batch256_cat7-checkpoint.h5
Epoch 3/50
4962550/4962550 [==============================] - 30527s 6ms/step - loss: 0.2970 - weighted_sparse_categorical_accuracy: 0.8965 - val_loss: 0.2728 - val_weighted_sparse_categorical_accuracy: 0.8994

Epoch 00003: val_weighted_sparse_categorical_accuracy improved from 0.89894 to 0.89945, saving model to bilstm128_batch256_cat7-checkpoint.h5
Epoch 4/50
4962550/4962550 [==============================] - 30363s 6ms/step - loss: 0.2804 - weighted_sparse_categorical_accuracy: 0.8984 - val_loss: 0.2801 - val_weighted_sparse_categorical_accuracy: 0.9002

Epoch 00004: val_weighted_sparse_categorical_accuracy improved from 0.89945 to 0.90024, saving model to bilstm128_batch256_cat7-checkpoint.h5
Epoch 5/50
4962550/4962550 [==============================] - 30385s 6ms/step - loss: 0.2858 - weighted_sparse_categorical_accuracy: 0.8989 - val_loss: 0.2728 - val_weighted_sparse_categorical_accuracy: 0.9002

Epoch 00005: val_weighted_sparse_categorical_accuracy did not improve from 0.90024
Epoch 6/50
4962550/4962550 [==============================] - 30462s 6ms/step - loss: 0.2743 - weighted_sparse_categorical_accuracy: 0.8998 - val_loss: 0.2653 - val_weighted_sparse_categorical_accuracy: 0.9019

[...]

4962550/4962550 [==============================] - 30703s 6ms/step - loss: 0.2697 - weighted_sparse_categorical_accuracy: 0.9006 - val_loss: 0.2636 - val_weighted_sparse_categorical_accuracy: 0.9021

Epoch 00008: val_weighted_sparse_categorical_accuracy improved from 0.90187 to 0.90208, saving model to bilstm128_batch256_cat7-checkpoint.h5


Per-class test measures:
1550797/1550797 [==============================] - 2751s 2ms/step

                 precision    recall  f1-score   support

acknowledgement       0.54      0.74      0.62      1013
      algorithm       0.65      0.54      0.59      2687
        caption       0.92      0.51      0.65       275
          proof       0.80      0.74      0.77    200000
     definition       0.84      0.80      0.82    164137
        problem       0.71      0.36      0.48      6629
          other       0.93      0.95      0.94   1176056

    avg / total       0.90      0.90      0.90   1550797

```

8. BiLSTM(128) + deep 1D-CNN, 256 batch, 1m max class

  - zero-rule baseline is 0.76 for class other(6)

```
Label summary:  {0: 5063, 1: 13435, 2: 1376, 3: 1000000, 4: 820683, 5: 33146, 6: 5880282}

6203188 train sequences
1550797 test sequences
x_train shape: (6203188, 480)
x_test shape: (1550797, 480)
y_train shape: (6203188,)
y_test shape: (1550797,)
loading word embeddings...
known dictionary items:  1000298
setting up model layout...
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 480, 300)          300089400 
_________________________________________________________________
dropout_1 (Dropout)          (None, 480, 300)          0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 480, 256)          439296    
_________________________________________________________________
dropout_2 (Dropout)          (None, 480, 256)          0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 240, 64)           81984     
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 120, 64)           0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 120, 64)           0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 120, 32)           6176      
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 60, 32)            0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 60, 32)            0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 60, 16)            1040      
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 30, 16)            0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 30, 16)            0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 480)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 7)                 3367      
=================================================================
Total params: 300,621,263
Trainable params: 531,863
Non-trainable params: 300,089,400
_________________________________________________________________
None
Training model...
Train on 4962550 samples, validate on 1240638 samples

4962550/4962550 [==============================] - 23930s 5ms/step - loss: 0.2668 - weighted_sparse_categorical_accuracy: 0.9020 - val_loss: 0.2659 - val_weighted_sparse_categorical_accuracy: 0.9028

Epoch 00005: val_weighted_sparse_categorical_accuracy did not improve from 0.90316
Saving model to disk : bilstm128_conv1d_batch256_cat7_1m 
Per-class test measures:
1550797/1550797 [==============================] - 2381s 2ms/step
                 precision    recall  f1-score   support

acknowledgement       0.57      0.06      0.11      1013
      algorithm       0.75      0.37      0.49      2687
        caption       0.99      0.34      0.51       275
          proof       0.78      0.79      0.78    200000
     definition       0.84      0.79      0.82    164137
        problem       0.68      0.36      0.47      6629
          other       0.93      0.94      0.94   1176056

    avg / total       0.90      0.90      0.90   1550797
```

9. BiLSTM(240) + Dense(960), batch 128, 5 classes, 1m class max
```
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  embedding_1 (Embedding)      (None, 480, 300)          300089400 
  _________________________________________________________________
  dropout_1 (Dropout)          (None, 480, 300)          0         
  _________________________________________________________________
  bidirectional_1 (Bidirection (None, 480)               1040640   
  _________________________________________________________________
  dropout_2 (Dropout)          (None, 480)               0         
  _________________________________________________________________
  dense_1 (Dense)              (None, 960)               461760    
  _________________________________________________________________
  dropout_3 (Dropout)          (None, 960)               0         
  _________________________________________________________________
  dense_2 (Dense)              (None, 5)                 4805      
  =================================================================
  Total params: 301,596,605
  Trainable params: 1,507,205
  Non-trainable params: 300,089,400

                  precision    recall  f1-score   support

acknowledgement   0.41      0.32      0.36      1013
          proof   0.76      0.78      0.77    200000
     definition   0.82      0.80      0.81    164137
        problem   0.64      0.39      0.48      6629
          other   0.93      0.94      0.93   1179018

    avg / total   0.90      0.90      0.90   1550797
```

10. BiLSTM(128), Dense(5, sigmoid), nadam, 1m max class
```
             precision    recall  f1-score   support

          0       0.52      0.86      0.65      1013
          1       0.80      0.74      0.77    200000
          2       0.81      0.81      0.81    164137
          3       0.65      0.38      0.48      6629
          4       0.93      0.94      0.94   1179018

avg / total       0.90      0.90      0.90   1550797
```
11. BiLSTM(128), Dense(5, softmax), adam, 1m max class
```
precision    recall  f1-score   support

          0       0.58      0.54      0.56      1013
          1       0.81      0.74      0.78    200000
          2       0.86      0.78      0.82    164137
          3       0.68      0.40      0.51      6629
          4       0.93      0.95      0.94   1179018

avg / total       0.90      0.91      0.90   1550797

```

11. v2 paragraph dataset, BiLSTM(128), Dense(5, softmax), adam, 1m max class
```
Label summary:  {0: 4786, 1: 1000000, 2: 693130, 3: 30089, 4: 5462069}

          0       0.64      0.13      0.22       957
          1       0.82      0.87      0.85    200000
          2       0.85      0.85      0.85    138626
          3       0.73      0.39      0.51      6018
          4       0.96      0.95      0.95   1092414

avg / total       0.93      0.92      0.92   1438015

```

12. v2 dataset,  BiLSTM(128), Dense(5, softmax), adam, 1m max class
```
  Saving model to disk : v2_bilstm128_batch256_cat23_gpu 
  Per-class test measures:
  1438015/1438015 [==============================] - 192s 134us/step
                    precision     recall f1-score   support
              proof   	0.82      	0.86  	0.84	  200000
         definition   	0.77      	0.91  	0.83	  138626
             remark   	0.64      	0.70  	0.67	  133607
              other   	0.68      	0.60  	0.64	  200000
            example   	0.71      	0.55  	0.62	  49741
            caption   	0.66      	0.51  	0.58	  270
            problem   	0.60      	0.49  	0.54	  6018
              lemma   	0.42      	0.57  	0.49	  200000
            theorem   	0.45      	0.53  	0.49	  200000
          algorithm   	0.48      	0.49  	0.48	  1480
    acknowledgement   	0.39      	0.42  	0.41	  957
         assumption   	0.51      	0.29  	0.37	  5776
           notation   	0.45      	0.25  	0.32	  10552
        proposition   	0.33      	0.31  	0.32	  184968
          corollary   	0.37      	0.04  	0.07	  87437
         conjecture   	0.64      	0.01  	0.01	  9082
               fact   	0.84      	0.01  	0.01	  3811
               case   	0.01      	0.00  	0.00	  721
          condition   	0.00	      0.00  	0.00	  798
          paragraph   	0.00	      0.00  	0.00	  405
           question   	0.00	      0.00  	0.00	  1667
             result   	0.00	      0.00  	0.00	  564
               step   	0.00	      0.00  	0.00	  1535
  avg / total          0.57        0.58      0.56   1438015
```

13. v2 data, BiLSTM(128), Dense(4, softmax), adam, 1m max -- NO OTHER
```
             precision    recall  f1-score   support

          0       0.99      1.00      0.99       957
          1       0.98      0.98      0.98    200000
          2       0.97      0.98      0.97    138626
          3       0.94      0.83      0.88      6018

avg / total       0.98      0.98      0.98    345601
```

Very promising, attempting a separate run with Other included, but all other named envs excluded.

14.  v2 data, BiLSTM(128), Dense(4, softmax), adam, 1m max -- Only explicit "Other"
```
          0       0.69      0.11      0.20       957
          1       0.89      0.92      0.90    200000
          2       0.89      0.95      0.92    138626
          3       0.85      0.78      0.81      6018
          4       0.89      0.82      0.86    200000

avg / total       0.89      0.89      0.89    545601
```

Can roughly conclude the "Other" class as such is too polluted with overlapping data... So, don't use it as such! Regenerate dataset where only explicitly *different* fixed classes are used from the rest of the document (e.g. abstract, introduction)
Then we can use a likelihood threshold to decide instead!!!

15. v2 data, BiLSTM(128), Dense(21, softmax), adam, 1m max -- Drop "other" only
Promising:
```
                  precision    recall  f1-score   support
acknowledgement   0.97      0.98      0.97       957
          proof   0.90      0.89      0.89    200000
     definition   0.84      0.91      0.88    138626
    proposition   0.76      0.79      0.77    133607
        example   0.70      0.65      0.67     49741
        caption   0.74      0.53      0.62       270
      algorithm   0.59      0.59      0.59      1480
        problem   0.64      0.53      0.58      6018
          lemma   0.41      0.67      0.51    200000
        theorem   0.47      0.48      0.48    200000

    avg / total   0.59      0.59      0.57   1238015
```

Full:
```
             precision    recall  f1-score   support

          0       0.97      0.98      0.97       957
          1       0.59      0.59      0.59      1480
          2       0.57      0.34      0.42      5776
          3       0.74      0.53      0.62       270
          4       0.35      0.10      0.15       721
          5       0.00      0.00      0.00       798
          6       0.49      0.01      0.02      9082
          7       0.36      0.05      0.09     87437
          8       0.84      0.91      0.88    138626
          9       0.70      0.65      0.67     49741
         10       0.57      0.01      0.01      3811
         11       0.41      0.67      0.51    200000
         12       0.45      0.29      0.36     10552
         13       0.50      0.00      0.00       405
         14       0.64      0.53      0.58      6018
         15       0.90      0.89      0.89    200000
         16       0.34      0.27      0.31    184968
         17       0.19      0.01      0.01      1667
         18       0.76      0.79      0.77    133607
         19       0.00      0.00      0.00       564
         20       0.56      0.00      0.01      1535
         21       0.47      0.48      0.48    200000

avg / total       0.59      0.59      0.57   1238015
```

16. v2 data, BiLSTM(128), Dense(6, softmax), adam, 1m max

With: acknowledgement(0), definition(1), example(2), lemma+theorem+proposition(3), problem(4), proof(5), NO other.

```
Label summary:  {0: 4786, 1: 693130, 2: 248706, 3: 2924838, 4: 30089, 5: 1000000}
Saving model to disk : v2_bilstm128_batch256_cat6_gpu 

                   precision    recall  f1-score   support

acknowledgement       0.98      0.99      0.99       957
     definition       0.93      0.90      0.92    138626
        example       0.83      0.71      0.76     49741
        theorem       0.95      0.98      0.97    584968
        problem       0.84      0.77      0.81      6018
          proof       0.96      0.93      0.94    200000

    avg / total       0.94      0.94      0.94    980310

```

17. v2 data, BiLSTM(128) + BiLSTM(64) + LSTM(32) + Dense(6, softmax), adam, 1m max
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 480, 300)          300089400 
_________________________________________________________________
dropout_1 (Dropout)          (None, 480, 300)          0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 480, 256)          440320    
_________________________________________________________________
dropout_2 (Dropout)          (None, 480, 256)          0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 480, 128)          164864    
_________________________________________________________________
dropout_3 (Dropout)          (None, 480, 128)          0         
_________________________________________________________________
cu_dnnlstm_3 (CuDNNLSTM)     (None, 64)                49664     
_________________________________________________________________
dropout_4 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 390       
=================================================================
Total params: 300,744,638
Trainable params: 655,238
Non-trainable params: 300,089,400

             precision    recall  f1-score   support

          0       0.98      0.99      0.99       957
          1       0.93      0.91      0.92    138626
          2       0.84      0.73      0.78     49741
          3       0.96      0.98      0.97    584968
          4       0.86      0.82      0.84      6018
          5       0.95      0.94      0.95    200000

avg / total       0.95      0.95      0.95    980310
```


17. v3 data, BiLSTM(128)

