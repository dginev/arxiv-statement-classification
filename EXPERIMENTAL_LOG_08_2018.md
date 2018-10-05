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
    [ongoing]
- BiLSTM(128) + BiLSTM(64), 7 classes (no assumption, no remark)
    [ongoing]


6. 250k, BiLSTM(120) x 2, batch 256
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
