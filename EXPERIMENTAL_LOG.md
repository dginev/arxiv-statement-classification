# Dense(300) baseline
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 300)               90300     
_________________________________________________________________
dropout_1 (Dropout)          (None, 300)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 9)                 2709      
=================================================================
Total params: 93,009
Trainable params: 93,009
Non-trainable params: 0
_________________________________________________________________
None
Evaluating model on test data...
sparse_categorical_accuracy: 66.82%
Saving model to disk...
Per-class test measures:
131551/131551 [==============================] - 2s 15us/step
  'precision', 'predicted', average, warn_for)
             precision    recall  f1-score   support

          0       0.00      0.00      0.00       862
          1       0.00      0.00      0.00      2717
          2       0.00      0.00      0.00       243
          3       0.11      0.00      0.00      9814
          4       0.00      0.00      0.00      4661
          5       0.24      0.00      0.00     10050
          6       0.00      0.00      0.00      5477
          7       0.12      0.00      0.00      9724
          8       0.67      1.00      0.80     88003

avg / total       0.48      0.67      0.54    131551
```

Deep Dense(600,300,150) baseline showed an identical result.

Learns the zero rule for Other, but that's about all.

# Dense(300) baseline + arXiv Glove embeddings
```
Evaluating model on test data...
sparse_categorical_accuracy: 71.18%
Saving model to disk...
Per-class test measures:
131551/131551 [==============================] - 7s 51us/step
             precision    recall  f1-score   support

          0       0.91      0.87      0.89       862
          1       0.75      0.53      0.62      2717
          2       0.87      0.37      0.52       243
          3       0.60      0.39      0.47      9814
          4       0.51      0.23      0.32      4661
          5       0.53      0.28      0.37     10050
          6       0.55      0.20      0.30      5477
          7       0.51      0.22      0.31      9724
          8       0.74      0.91      0.82     88003

avg / total       0.68      0.71      0.68    131551
```

# CNN(mix-depth 3,5,18,33)
```
             precision    recall  f1-score   support

          0       0.93      0.99      0.96       862
          1       0.83      0.71      0.77      2717
          2       0.80      0.63      0.71       243
          3       0.67      0.62      0.64      9814
          4       0.70      0.34      0.46      4661
          5       0.67      0.56      0.61     10050
          6       0.83      0.46      0.59      5477
          7       0.71      0.25      0.37      9724
          8       0.79      0.92      0.85     88003

avg / total       0.77      0.78      0.76    131551
```

# CNN(100f,2k)
```_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 300, 300)          224002800 
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 299, 100)          60100     
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 100)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               25856     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 9)                 2313      
=================================================================
Total params: 224,091,069
Trainable params: 88,269
Non-trainable params: 224,002,800
_________________________________________________________________
None
Train on 420962 samples, validate on 105241 samples
Epoch 1/5
420962/420962 [==============================] - 20s 47us/step - loss: 0.7392 - sparse_categorical_accuracy: 0.7392 - val_loss: 0.6643 - val_sparse_categorical_accuracy: 0.7611
Epoch 2/5
420962/420962 [==============================] - 19s 44us/step - loss: 0.6555 - sparse_categorical_accuracy: 0.7623 - val_loss: 0.6536 - val_sparse_categorical_accuracy: 0.7629
Epoch 3/5
420962/420962 [==============================] - 19s 44us/step - loss: 0.6323 - sparse_categorical_accuracy: 0.7701 - val_loss: 0.6357 - val_sparse_categorical_accuracy: 0.7689
Epoch 4/5
420962/420962 [==============================] - 19s 44us/step - loss: 0.6182 - sparse_categorical_accuracy: 0.7744 - val_loss: 0.6294 - val_sparse_categorical_accuracy: 0.7700
Epoch 5/5
420962/420962 [==============================] - 19s 44us/step - loss: 0.6070 - sparse_categorical_accuracy: 0.7777 - val_loss: 0.6311 - val_sparse_categorical_accuracy: 0.7694
Evaluating model on test data...
sparse_categorical_accuracy: 77.09%
Saving model to disk...
Per-class test measures:
131551/131551 [==============================] - 3s 25us/step
             precision    recall  f1-score   support

          0       0.90      0.99      0.94       862
          1       0.81      0.70      0.75      2717
          2       0.70      0.60      0.65       243
          3       0.66      0.59      0.62      9814
          4       0.70      0.33      0.45      4661
          5       0.70      0.42      0.53     10050
          6       0.75      0.50      0.60      5477
          7       0.63      0.36      0.46      9724
          8       0.79      0.92      0.85     88003

avg / total       0.76      0.77      0.75    131551
```

# CNN(150f, ngram kernel - 2,3,4,5,6,8)
```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 300)          0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 300, 300)     224002800   input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 299, 150)     90150       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 298, 150)     135150      embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 297, 150)     180150      embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 296, 150)     225150      embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 295, 150)     270150      embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, 293, 150)     360150      embedding_1[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 150)          0           conv1d_1[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 150)          0           conv1d_2[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalM (None, 150)          0           conv1d_3[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_4 (GlobalM (None, 150)          0           conv1d_4[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_5 (GlobalM (None, 150)          0           conv1d_5[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_6 (GlobalM (None, 150)          0           conv1d_6[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 900)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 global_max_pooling1d_3[0][0]     
                                                                 global_max_pooling1d_4[0][0]     
                                                                 global_max_pooling1d_5[0][0]     
                                                                 global_max_pooling1d_6[0][0]     
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 900)          0           concatenate_1[0][0]              
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 900)          810900      dropout_1[0][0]                  
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 900)          0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 9)            8109        dropout_2[0][0]                  
==================================================================================================
Total params: 226,082,709
Trainable params: 2,079,909
Non-trainable params: 224,002,800
__________________________________________________________________________________________________
None
Train on 420962 samples, validate on 105241 samples
Epoch 1/5
420962/420962 [==============================] - 171s 407us/step - loss: 0.7709 - sparse_categorical_accuracy: 0.7312 
                                                             - val_loss: 0.6712 - val_sparse_categorical_accuracy: 0.7622
Epoch 2/5
420962/420962 [==============================] - 169s 400us/step - loss: 0.6869 - sparse_categorical_accuracy: 0.7533 
                                                             - val_loss: 0.6430 - val_sparse_categorical_accuracy: 0.7660
Epoch 3/5
420962/420962 [==============================] - 169s 400us/step - loss: 0.6659 - sparse_categorical_accuracy: 0.7596 
                                                             - val_loss: 0.6294 - val_sparse_categorical_accuracy: 0.7742
Epoch 4/5
420962/420962 [==============================] - 170s 403us/step - loss: 0.6504 - sparse_categorical_accuracy: 0.7631 
                                                             - val_loss: 0.6325 - val_sparse_categorical_accuracy: 0.7678
Epoch 5/5
420962/420962 [==============================] - 166s 395us/step - loss: 0.6404 - sparse_categorical_accuracy: 0.7671 
                                                             - val_loss: 0.6223 - val_sparse_categorical_accuracy: 0.7778
Evaluating model on test data...
sparse_categorical_accuracy: 77.83%
```

# biLSTM(32), sgd optimizer, dense softmax run:
First ever reasonable try

```
 _________________________________________________________________
 Layer (type)                 Output Shape              Param #
 =================================================================
 embedding_1 (Embedding)      (None, 150, 300)          224002800
 _________________________________________________________________
 bidirectional_1 (Bidirection (None, 64)                85248
 _________________________________________________________________
 dropout_1 (Dropout)          (None, 64)                0
 _________________________________________________________________
 dense_1 (Dense)              (None, 23)                1495
 =================================================================
 Total params: 224,089,543
 Trainable params: 86,743

 Train on 411703 samples, validate on 102926 samples
 Epoch 1/2
 411703/411703 [==============================] - 2690s 7ms/step - loss: 2.3600 - sparse_categorical_accuracy: 0.2427
                                                             - val_loss: 2.1382 - val_sparse_categorical_accuracy: 0.3077
 Epoch 2/2
 411703/411703 [==============================] - 2702s 7ms/step - loss: 2.0567 - sparse_categorical_accuracy: 0.3320 
                                                             - val_loss: 1.9705 - val_sparse_categorical_accuracy: 0.3538
 Evaluating model on test data...
 sparse_categorical_accuracy: 35.39%
 
```

# biLSTM(64), adam, dense softmax run:

```
 Train on 411703 samples, validate on 102926 samples
 Epoch 1/10
 411703/411703 [==============================] - 1767s 4ms/step - loss: 1.9364 - sparse_categorical_accuracy: 0.3636
                                                             - val_loss: 1.7679 - val_sparse_categorical_accuracy: 0.4110
 Epoch 2/10
 411703/411703 [==============================] - 1769s 4ms/step - loss: 1.7436 - sparse_categorical_accuracy: 0.4175
                                                             - val_loss: 1.7112 - val_sparse_categorical_accuracy: 0.4269
 Epoch 3/10
 411703/411703 [==============================] - 1762s 4ms/step - loss: 1.6856 - sparse_categorical_accuracy: 0.4326
                                                             - val_loss: 1.6803 - val_sparse_categorical_accuracy: 0.4367
 Epoch 4/10
 411703/411703 [==============================] - 1756s 4ms/step - loss: 1.6503 - sparse_categorical_accuracy: 0.4429
                                                             - val_loss: 1.6722 - val_sparse_categorical_accuracy: 0.4376
 Epoch 5/10
 411703/411703 [==============================] - 1751s 4ms/step - loss: 1.6208 - sparse_categorical_accuracy: 0.4514
                                                             - val_loss: 1.6645 - val_sparse_categorical_accuracy: 0.4387
 Epoch 10/10
 411703/411703 [==============================] - 1749s 4ms/step - loss: 1.5280 - sparse_categorical_accuracy: 0.4773
                                                             - val_loss: 1.6685 - val_sparse_categorical_accuracy: 0.4414
 Evaluating model on test data...
 sparse_categorical_accuracy: 44.05%
```

# biLSTM(75), adam, strict labels, dense softmax run:

```
 _________________________________________________________________
 Layer (type)                 Output Shape              Param #
 =================================================================
 embedding_1 (Embedding)      (None, 150, 300)          224002800
 _________________________________________________________________
 bidirectional_1 (Bidirection (None, 150)               225600
 _________________________________________________________________
 dropout_1 (Dropout)          (None, 150)               0
 _________________________________________________________________
 dense_1 (Dense)              (None, 11)                1661
 =================================================================
 
 Total params: 224,230,061
 Trainable params: 227,261
 Non-trainable params: 224,002,800
 _________________________________________________________________
 None
 Train on 411703 samples, validate on 102926 samples
 Epoch 1/3
 411703/411703 [==============================] - 1781s 4ms/step - loss: 1.3392 - sparse_categorical_accuracy: 0.5091
                                                             - val_loss: 1.2083 - val_sparse_categorical_accuracy: 0.5503
 Epoch 2/3
 411703/411703 [==============================] - 1760s 4ms/step - loss: 1.1929 - sparse_categorical_accuracy: 0.5538
                                                             - val_loss: 1.1803 - val_sparse_categorical_accuracy: 0.5590
 Epoch 3/3
 411703/411703 [==============================] - 1781s 4ms/step - loss: 1.1484 - sparse_categorical_accuracy: 0.5678
                                                             - val_loss: 1.1557 - val_sparse_categorical_accuracy: 0.5682
 Evaluating model on test data...
 sparse_categorical_accuracy: 56.57%
```

# biLSTM(150), adam, strict labels, dense softmax run:
Conclusion: LSTM 75 seems good enough? for the strict labels

```_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 150, 300)          224002800 
_________________________________________________________________
bidirectional_1 (Bidirection (None, 300)               541200    
_________________________________________________________________
dropout_1 (Dropout)          (None, 300)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 11)                3311      
=================================================================

Total params: 224,547,311
Trainable params: 544,511
Non-trainable params: 224,002,800
_________________________________________________________________
None
Train on 411703 samples, validate on 102926 samples
Epoch 1/3
411703/411703 [==============================] - 1766s 4ms/step - loss: 1.3028 - sparse_categorical_accuracy: 0.5187
                                                            - val_loss:  1.1837 - val_sparse_categorical_accuracy: 0.5583
Epoch 2/3
411703/411703 [==============================] - 1762s 4ms/step - loss: 1.1525 - sparse_categorical_accuracy: 0.5655 
                                                            - val_loss: 1.1442 - val_sparse_categorical_accuracy: 0.5690
Epoch 3/3
411703/411703 [==============================] - 1770s 4ms/step - loss: 1.1018 - sparse_categorical_accuracy: 0.5823 
                                                            - val_loss: 1.1249 - val_sparse_categorical_accuracy: 0.5767
Evaluating model on test data...
sparse_categorical_accuracy: 57.48%
```

# biLSTM(75), adam, strict labels, dense sigmoid run:
Not better, sticking to softmax
```
411703/411703 [==============================] - 1805s 4ms/step - loss: 1.3572 - sparse_categorical_accuracy: 0.5024
                                                            - val_loss: 1.2154 - val_sparse_categorical_accuracy: 0.5485

```
# biLSTM(75)+biLSTM(75), (adam, strict labels, dense softmax) run:

```
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 150, 300)          224002800 
_________________________________________________________________
dropout_1 (Dropout)          (None, 150, 300)          0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 150, 150)          225600    
_________________________________________________________________
dropout_2 (Dropout)          (None, 150, 150)          0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 150)               135600    
_________________________________________________________________
dropout_3 (Dropout)          (None, 150)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 11)                1661      
=================================================================
Total params: 224,365,661
Trainable params: 362,861
Non-trainable params: 224,002,800

Epoch 1/3
411703/411703 [==============================] - 3560s 9ms/step - loss: 1.3368 - sparse_categorical_accuracy: 0.5095 
                                                            - val_loss: 1.1931 - val_sparse_categorical_accuracy: 0.5562
[stopped]
```

# biLSTM(7*n_classes)+biLSTM(7*n_classes)+Dense(3*n_classes), batch size = 100
## strict labels

```_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 150, 300)          224002800 
_________________________________________________________________
dropout_1 (Dropout)          (None, 150, 300)          0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 150, 154)          232848    
_________________________________________________________________
dropout_2 (Dropout)          (None, 150, 154)          0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 154)               142912    
_________________________________________________________________
dropout_3 (Dropout)          (None, 154)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 33)                5115      
_________________________________________________________________
dropout_4 (Dropout)          (None, 33)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 11)                374       
=================================================================
Total params: 224,384,049
Trainable params: 381,249
Non-trainable params: 224,002,800
_________________________________________________________________
None
Train on 411703 samples, validate on 102926 samples
Epoch 1/3
411703/411703 [==============================] - 3684s 9ms/step - loss: 1.3837 - sparse_categorical_accuracy: 0.4989 -
                                                              val_loss: 1.2073 - val_sparse_categorical_accuracy: 0.5537
Epoch 2/3
411703/411703 [==============================] - 3712s 9ms/step - loss: 1.2141 - sparse_categorical_accuracy: 0.5525
                                                            - val_loss: 1.1624 - val_sparse_categorical_accuracy: 0.5647

Epoch 3/3
411703/411703 [==============================] - 3684s 9ms/step - loss: 1.1721 - sparse_categorical_accuracy: 0.5650 - 
                                                              val_loss: 1.1330 - val_sparse_categorical_accuracy: 0.5741

Evaluating model on test data...
sparse_categorical_accuracy: 57.15%
```

## full labels

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 150, 300)          224002800 
_________________________________________________________________
dropout_1 (Dropout)          (None, 150, 300)          0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 150, 322)          595056    
_________________________________________________________________
dropout_2 (Dropout)          (None, 150, 322)          0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 322)               623392    
_________________________________________________________________
dropout_3 (Dropout)          (None, 322)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 69)                22287     
_________________________________________________________________
dropout_4 (Dropout)          (None, 69)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 23)                1610      
=================================================================
Total params: 225,245,145
Trainable params: 1,242,345
Non-trainable params: 224,002,800
_________________________________________________________________
None
Train on 411703 samples, validate on 102926 samples
Epoch 1/3
411703/411703 [==============================] - 3669s 9ms/step - loss: 1.9224 - sparse_categorical_accuracy: 0.3680 
                                                            - val_loss: 1.7160 - val_sparse_categorical_accuracy: 0.4249
Epoch 2/3
411703/411703 [==============================] - 3688s 9ms/step - loss: 1.7145 - sparse_categorical_accuracy: 0.4278 
                                                            - val_loss: 1.6625 - val_sparse_categorical_accuracy: 0.4409

Epoch 3/3
411703/411703 [==============================] - 3666s 9ms/step - loss: 1.6510 - sparse_categorical_accuracy: 0.4446
                                                            - val_loss: 1.6311 - val_sparse_categorical_accuracy: 0.4478

Evaluating model on test data...
sparse_categorical_accuracy: 44.74%
```

# biLSTM(2*maxlen)+biLSTM(2*maxlen), batch size = 128, strict labels, 0.2 dropout

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 150, 300)          224002800 
_________________________________________________________________
bidirectional_1 (Bidirection (None, 150, 600)          1442400   
_________________________________________________________________
dropout_1 (Dropout)          (None, 150, 600)          0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 600)               2162400   
_________________________________________________________________
dropout_2 (Dropout)          (None, 600)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 11)                6611      
=================================================================
Total params: 227,614,211
Trainable params: 3,611,411
Non-trainable params: 224,002,800
_________________________________________________________________

Epoch 1/2
411703/411703 [==============================] - 2833s 7ms/step - loss: 1.2634 - sparse_categorical_accuracy: 0.5302
                                               - val_loss: 1.1509 - val_sparse_categorical_accuracy: 0.5680

Epoch 2/2
411703/411703 [==============================] - 2828s 7ms/step - loss: 1.1111 - sparse_categorical_accuracy: 0.5785 
                                                            - val_loss: 1.1067 - val_sparse_categorical_accuracy: 0.5800
Evaluating model on test data...
sparse_categorical_accuracy: 57.83%

```

## Per-class test measures (strict labels):
```
128658/128658 [==============================] - 1037s 8ms/step
             precision    recall  f1-score   support

1 definition 0.65      0.74      0.69     12531
2 example    0.56      0.53      0.54      9511
3 notation   0.56      0.37      0.45      9818
4 problem    0.61      0.69      0.64      5453
5 proof      0.63      0.71      0.67      9270
6 propositn. 0.56      0.84      0.67     37708
7 question   0.29      0.08      0.12      1376
8 remark     0.56      0.53      0.54     12016
9 theorem    0.51      0.18      0.27     20012
10 other     0.64      0.39      0.49     10963

avg / total       0.57      0.58      0.55    128658

```
        
# biLSTM(maxlen)+biLSTM(maxlen), batch size = 128, 0.2 dropout

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 150, 300)          224002800 
_________________________________________________________________
bidirectional_1 (Bidirection (None, 150, 300)          541200    
_________________________________________________________________
dropout_1 (Dropout)          (None, 150, 300)          0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 300)               541200    
_________________________________________________________________
dropout_2 (Dropout)          (None, 300)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 23)                6923      
=================================================================
Total params: 225,092,123
Trainable params: 1,089,323
Non-trainable params: 224,002,800
_________________________________________________________________
None
Train on 411703 samples, validate on 102926 samples
Epoch 1/2
411703/411703 [==============================] - 2894s 7ms/step - loss: 1.8683 - sparse_categorical_accuracy: 0.3802 
                                                            - val_loss: 1.7048 - val_sparse_categorical_accuracy: 0.4291
Epoch 2/2
411703/411703 [==============================] - 2909s 7ms/step - loss: 1.6637 - sparse_categorical_accuracy: 0.4380 
                                                            - val_loss: 1.6422 - val_sparse_categorical_accuracy: 0.4435
Evaluating model on test data...
sparse_categorical_accuracy: 44.21%

```

## Per-class test measures (full labels):
```
Per-class test measures:
             precision    recall  f1-score   support

          0       0.94      0.98      0.96       858
          1       0.81      0.70      0.75      2593
          2       0.50      0.63      0.56      4641
          3       0.81      0.60      0.69       240
          4       0.44      0.15      0.23       837
          5       0.81      0.03      0.05       657
          6       0.36      0.35      0.35      7989
          7       0.29      0.23      0.26     10104
          8       0.56      0.77      0.65      9938
          9       0.53      0.54      0.53      9511
         10       0.31      0.19      0.23      3526
         11       0.30      0.23      0.26     10097
         12       0.49      0.51      0.50      9818
         13       0.39      0.40      0.39      9224
         14       0.64      0.06      0.10       641
         15       0.79      0.51      0.62      5453
         16       0.61      0.73      0.66      9270
         17       0.21      0.39      0.28      9954
         18       0.23      0.01      0.02      1376
         19       0.53      0.60      0.57      9380
         20       0.25      0.00      0.00       553
         21       0.34      0.12      0.17      2083
         22       0.34      0.21      0.26      9915

avg / total       0.44      0.44      0.43    128658
```

## Dropping `Other` class
```
Train on 382157 samples, validate on 95540 samples
Epoch 1/2
382157/382157 [==============================] - 2666s 7ms/step - loss: 1.7980 - sparse_categorical_accuracy: 0.3975
                                                            - val_loss: 1.6284 - val_sparse_categorical_accuracy: 0.4455
Epoch 2/2
382157/382157 [==============================] - 2639s 7ms/step - loss: 1.5923 - sparse_categorical_accuracy: 0.4549
                                                            - val_loss: 1.5718 - val_sparse_categorical_accuracy: 0.4625
Evaluating model on test data...
sparse_categorical_accuracy: 46.11%

             precision    recall  f1-score   support

          0       0.99      0.99      0.99       858
          1       0.76      0.75      0.75      2593
          2       0.49      0.65      0.56      4641
          3       0.77      0.65      0.70       240
          4       0.41      0.21      0.28       837
          5       0.45      0.01      0.01       657
          6       0.36      0.38      0.37      7988
          7       0.29      0.23      0.26     10102
          8       0.60      0.72      0.65      9938
          9       0.48      0.55      0.51      9511
         10       0.34      0.18      0.24      3526
         11       0.28      0.45      0.34     10097
         12       0.46      0.58      0.51      9818
         13       0.39      0.08      0.13       641
         14       0.74      0.54      0.62      5453
         15       0.66      0.75      0.70      9270
         16       0.24      0.14      0.17      9954
         17       0.30      0.01      0.01      1376
         18       0.60      0.67      0.63      9378
         19       0.00      0.00      0.00       553
         20       0.34      0.14      0.20      2082
         21       0.32      0.25      0.28      9912

avg / total       0.45      0.46      0.44    119425

```

# BiLSTM(maxlen)+BiLSTM(maxlen) with "10 best f1"-classes

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 150, 300)          224002800 
_________________________________________________________________
bidirectional_1 (Bidirection (None, 150, 300)          541200    
_________________________________________________________________
dropout_1 (Dropout)          (None, 150, 300)          0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 300)               541200    
_________________________________________________________________
dropout_2 (Dropout)          (None, 300)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 11)                3311      
=================================================================
Total params: 225,088,511
Trainable params: 1,085,711
Non-trainable params: 224,002,800
_________________________________________________________________

Train on 198123 samples, validate on 49531 samples
Epoch 1/2
198123/198123 [==============================] - 1363s 7ms/step - loss: 1.0868 - sparse_categorical_accuracy: 0.5996 
                                                            - val_loss: 0.9214 - val_sparse_categorical_accuracy: 0.6617
Epoch 2/2
198123/198123 [==============================] - 1346s 7ms/step - loss: 0.8849 - sparse_categorical_accuracy: 0.6731 
                                                            - val_loss: 0.8629 - val_sparse_categorical_accuracy: 0.6830
Evaluating model on test data...
sparse_categorical_accuracy: 68.01%

Per-class test measures:
61914/61914 [==============================] - 510s 8ms/step
             precision    recall  f1-score   support

          0       0.99      1.00      0.99       863
          1       0.78      0.80      0.79      2602
          2       0.85      0.68      0.75       242
          3       0.76      0.74      0.75      9298
          4       0.71      0.82      0.76      4659
          5       0.66      0.78      0.71      9963
          6       0.63      0.58      0.61      9543
          7       0.59      0.53      0.56      9863
          8       0.76      0.57      0.65      5470
          9       0.65      0.71      0.68      9411

avg / total       0.68      0.68      0.68     61914

```

## 9 f1-best classes

```
Train on 166560 samples, validate on 41641 samples
Epoch 1/3
166560/166560 [==============================] - 1168s 7ms/step - loss: 0.9267 - sparse_categorical_accuracy: 0.6690
                                                            - val_loss: 0.7516 - val_sparse_categorical_accuracy: 0.7354
Epoch 2/3
166560/166560 [==============================] - 1158s 7ms/step - loss: 0.7200 - sparse_categorical_accuracy: 0.7454 
                                                            - val_loss: 0.6904 - val_sparse_categorical_accuracy: 0.7561
Epoch 3/3
166560/166560 [==============================] - 1169s 7ms/step - loss: 0.6499 - sparse_categorical_accuracy: 0.7700 
                                                            - val_loss: 0.6757 - val_sparse_categorical_accuracy: 0.7655

Evaluating model on test data...
sparse_categorical_accuracy: 76.30%

Per-class test measures:
52051/52051 [==============================] - 420s 8ms/step
                 precision    recall  f1-score   support
      
acknowledgement  0       0.99      1.00      0.99       863
algorithm        1       0.83      0.77      0.80      2602
caption          2       0.73      0.72      0.72       242
proof            3       0.74      0.80      0.77      9298
assumption       4       0.77      0.82      0.79      4659
definition       5       0.85      0.86      0.85      9963
problem          6       0.66      0.59      0.62      9543
remark           7       0.88      0.83      0.85      5470
example          8       0.69      0.72      0.70      9411

avg / total       0.76      0.76      0.76     52051


```

## 8 f1-best classes

```
Train on 136392 samples, validate on 34098 samples

Epoch 1/3
136392/136392 [==============================] - 957s 7ms/step - loss: 0.7353 - sparse_categorical_accuracy: 0.7406
                                                           - val_loss: 0.5619 - val_sparse_categorical_accuracy: 0.8039
Epoch 2/3
136392/136392 [==============================] - 948s 7ms/step - loss: 0.5385 - sparse_categorical_accuracy: 0.8121 
                                                           - val_loss: 0.5176 - val_sparse_categorical_accuracy: 0.8188
Epoch 3/3
136392/136392 [==============================] - 952s 7ms/step - loss: 0.4753 - sparse_categorical_accuracy: 0.8333 
                                                           - val_loss: 0.4958 - val_sparse_categorical_accuracy: 0.8291

Evaluating model on test data...
sparse_categorical_accuracy: 82.90%

Per-class test measures:
42623/42623 [==============================] - 347s 8ms/step
                    precision    recall  f1-score   support

 acknowledgement 0       0.99      1.00      1.00       864
 algorithm       1       0.80      0.82      0.81      2609
 caption         2       0.73      0.74      0.74       243
 proof           3       0.81      0.82      0.81      9324
 assumption      4       0.79      0.83      0.81      4674
 definition      5       0.85      0.89      0.87      9987
 problem         6       0.91      0.82      0.86      5487
 remark          7       0.80      0.77      0.79      9435

avg / total       0.83      0.83      0.83     42623

```

## 8 f1-best classes + other class

```
Train on 411703 samples, validate on 102926 samples
Epoch 1/3
411703/411703 [==============================] - 2822s 7ms/step - loss: 0.7073 - sparse_categorical_accuracy: 0.7478 
                                                            - val_loss: 0.6183 - val_sparse_categorical_accuracy: 0.7762
Epoch 2/3
411703/411703 [==============================] - 2812s 7ms/step - loss: 0.5966 - sparse_categorical_accuracy: 0.7830 
                                                            - val_loss: 0.5890 - val_sparse_categorical_accuracy: 0.7846
Epoch 3/3
411703/411703 [==============================] - 2792s 7ms/step - loss: 0.5588 - sparse_categorical_accuracy: 0.7949 
                                                            - val_loss: 0.5770 - val_sparse_categorical_accuracy: 0.7888

Evaluating model on test data...
sparse_categorical_accuracy: 79.09%

Per-class test measures:
128658/128658 [==============================] - 1055s 8ms/step
             precision    recall  f1-score   support

          0       0.93      0.99      0.96       858
          1       0.87      0.67      0.76      2593
          2       0.89      0.67      0.76       240
          3       0.69      0.63      0.66      9270
          4       0.70      0.42      0.53      4641
          5       0.67      0.66      0.66      9938
          6       0.81      0.51      0.63      5453
          7       0.59      0.56      0.57      9380
          8       0.83      0.89      0.86     86285

avg / total       0.79      0.79      0.79    128658
```


# Full Dataset Training 

## EXPERIMENT I: 2x150 BiLSTM
(cap Other class at 5 million entries, for performance)
Total Input; Label summary:  {0: 4216, 1: 13741, 2: 1369, 3: 4213367, 4: 23074, 5: 675025, 6: 27482, 7: 644450, 8: 9137806}

```
performing train/test cutoff at index  11224802 / 14031003 ...
11224802 train sequences
2806201 test sequences

Training model...
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 150, 300)          224002800 
_________________________________________________________________
bidirectional_1 (Bidirection (None, 150, 300)          541200    
_________________________________________________________________
dropout_1 (Dropout)          (None, 150, 300)          0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 300)               541200    
_________________________________________________________________
dropout_2 (Dropout)          (None, 300)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 9)                 2709      
=================================================================
Total params: 225,087,909
Trainable params: 1,085,109
Non-trainable params: 224,002,800

Train on 8979841 samples, validate on 2244961 samples
8979840/8979841 [============================>.] - ETA: 0s - loss: 0.4671 - sparse_categorical_accuracy: 0.8265
                                                       - val_loss: 0.4425 - val_sparse_categorical_accuracy: 0.8347

Epoch 2/10
8979841/8979841 [==============================] - 60209s 7ms/step - loss: 0.4420 - sparse_categorical_accuracy: 0.8352 
                                                               - val_loss: 0.4356 - val_sparse_categorical_accuracy: 0.8370
Epoch 3/10
4694272/8979841 [==============>...............] - ETA: 7:21:46 - loss: 0.4362 - sparse_categorical_accuracy: 0.8373 

[interrupted due to early saturation and slow runtime]
```

## EXPERIMENT II: 150 BiLSTM, 300 maxlen
(cap classes at 500,000)

```
Label summary:  {0: 4216, 1: 13741, 2: 1369, 3: 500000, 4: 23074, 5: 500000, 6: 27482, 7: 500000, 8: 2870700}

performing train/test cutoff at index  3543062 / 4428828 ...
3543062 train sequences
885766 test sequences

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 300, 300)          224002800 
_________________________________________________________________
bidirectional_1 (Bidirection (None, 300)               541200    
_________________________________________________________________
dropout_1 (Dropout)          (None, 300)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 9)                 2709      
=================================================================
Total params: 224,546,709
Trainable params: 543,909
Non-trainable params: 224,002,800
_________________________________________________________________
None
Train on 2834449 samples, validate on 708613 samples
Epoch 1/10
2834449/2834449 [==============================] - 19895s 7ms/step - loss: 0.5282 - sparse_categorical_accuracy: 0.8106 
                                                               - val_loss: 0.4942 - val_sparse_categorical_accuracy: 0.8222
Epoch 2/10
2834449/2834449 [==============================] - 19860s 7ms/step - loss: 0.4810 - sparse_categorical_accuracy: 0.8279
                                                               - val_loss: 0.4822 - val_sparse_categorical_accuracy: 0.8270
Epoch 3/10
2834449/2834449 [==============================] - 19748s 7ms/step - loss: 0.4668 - sparse_categorical_accuracy: 0.8326 
                                                               - val_loss: 0.4752 - val_sparse_categorical_accuracy: 0.8290

Epoch 4/10
2834449/2834449 [==============================] - 19437s 7ms/step - loss: 0.4572 - sparse_categorical_accuracy: 0.8359 
                                                               - val_loss: 0.4749 - val_sparse_categorical_accuracy: 0.8295

Epoch 5/10
2834449/2834449 [==============================] - 19732s 7ms/step - loss: 0.4501 - sparse_categorical_accuracy: 0.8384
                                                               - val_loss: 0.4769 - val_sparse_categorical_accuracy: 0.8296

Epoch 6/10
2834449/2834449 [==============================] - 19972s 7ms/step - loss: 0.4444 - sparse_categorical_accuracy: 0.8405 
                                                               - val_loss: 0.4763 - val_sparse_categorical_accuracy: 0.8291

Epoch 7/10
2834449/2834449 [==============================] - 19571s 7ms/step - loss: 0.4396 - sparse_categorical_accuracy: 0.8423 
                                                               - val_loss: 0.4753 - val_sparse_categorical_accuracy: 0.8303

Epoch 8/10
2834449/2834449 [==============================] - 19543s 7ms/step - loss: 0.4357 - sparse_categorical_accuracy: 0.8437
                                                               - val_loss: 0.4773 - val_sparse_categorical_accuracy: 0.8288

Epoch 9/10
2834449/2834449 [==============================] - 19496s 7ms/step - loss: 0.4327 - sparse_categorical_accuracy: 0.8446
                                                               - val_loss: 0.4766 - val_sparse_categorical_accuracy: 0.8298

Epoch 10/10
2834449/2834449 [==============================] - 20089s 7ms/step - loss: 0.4301 - sparse_categorical_accuracy: 0.8455
                                                               - val_loss: 0.4781 - val_sparse_categorical_accuracy: 0.8289

Evaluating model on test data...
sparse_categorical_accuracy: 82.96%
Saving model to disk... (model-150-f1-9-classes-big.h5)

Per-class test measures:
885766/885766 [==============================] - 7399s 8ms/step
                 precision    recall  f1-score   support

 acknowledgement 0       0.66      0.70      0.68       861
 algorithm       1       0.75      0.52      0.61      2649
 caption         2       0.77      0.51      0.61       267
 proof           3       0.75      0.74      0.74     99251
 assumption      4       0.66      0.18      0.28      4640
 definition      5       0.82      0.79      0.81    100324
 problem         6       0.67      0.46      0.55      5570
 remark          7       0.69      0.59      0.64     99476
 other           8       0.86      0.90      0.88    572728

 avg / total       0.83      0.83      0.83    885766

```

Discussion: 3 epochs seem sufficient.

## EXPERIMENT III: 2x150BiLSTM, 300 maxlen
(cap classes at 500,000)

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 300, 300)          224002800 
_________________________________________________________________
bidirectional_1 (Bidirection (None, 300, 300)          541200    
_________________________________________________________________
dropout_1 (Dropout)          (None, 300, 300)          0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 300)               541200    
_________________________________________________________________
dropout_2 (Dropout)          (None, 300)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 9)                 2709      
=================================================================
Total params: 225,087,909
Trainable params: 1,085,109
Non-trainable params: 224,002,800
_________________________________________________________________
None
Train on 2834449 samples, validate on 708613 samples
Epoch 1/10
2834449/2834449 [==============================] - 39888s 14ms/step - loss: 0.5172 - sparse_categorical_accuracy: 0.8143
                                                                - val_loss: 0.4749 - val_sparse_categorical_accuracy: 0.8296
Epoch 2/10
2834449/2834449 [==============================] - 39322s 14ms/step - loss: 0.4664 - sparse_categorical_accuracy: 0.8328
                                                                - val_loss: 0.4622 - val_sparse_categorical_accuracy: 0.8338
Epoch 3/10
2834449/2834449 [==============================] - 39773s 14ms/step - loss: 0.4514 - sparse_categorical_accuracy: 0.8381
                                                                - val_loss: 0.4585 - val_sparse_categorical_accuracy: 0.8351
Epoch 4/10
2834449/2834449 [==============================] - 39171s 14ms/step - loss: 0.4421 - sparse_categorical_accuracy: 0.8415
                                                                - val_loss: 0.4562 - val_sparse_categorical_accuracy: 0.8364
Epoch 5/10
2834449/2834449 [==============================] - 39614s 14ms/step - loss: 0.4356 - sparse_categorical_accuracy: 0.8440 
                                                                - val_loss: 0.4559 - val_sparse_categorical_accuracy: 0.8370
Epoch 6/10
2834449/2834449 [==============================] - 39141s 14ms/step - loss: 0.4310 - sparse_categorical_accuracy: 0.8455
                                                                - val_loss: 0.4569 - val_sparse_categorical_accuracy: 0.8364
Epoch 7/10
2834449/2834449 [==============================] - 38586s 14ms/step - loss: 0.4276 - sparse_categorical_accuracy: 0.8467 
                                                                - val_loss: 0.4591 - val_sparse_categorical_accuracy: 0.8358
Epoch 8/10                                                                
2834449/2834449 [==============================] - 38865s 14ms/step - loss: 0.4250 - sparse_categorical_accuracy: 0.8477
                                                                - val_loss: 0.4592 - val_sparse_categorical_accuracy: 0.8358

Epoch 9/10
2834449/2834449 [==============================] - 38611s 14ms/step - loss: 0.4233 - sparse_categorical_accuracy: 0.8480 
                                                                - val_loss: 0.4583 - val_sparse_categorical_accuracy: 0.8361

Epoch 10/10
2834449/2834449 [==============================] - 38882s 14ms/step - loss: 0.4220 - sparse_categorical_accuracy: 0.8487 
                                                                - val_loss: 0.4603 - val_sparse_categorical_accuracy: 0.8364
Evaluating model on test data...
sparse_categorical_accuracy: 83.69%
Saving model to disk... (model-2x150-f1-9-classes-big.h5)
Per-class test measures:
885766/885766 [==============================] - 14211s 16ms/step
                precision    recall  f1-score   support

acknowledgement 0       0.66      0.83      0.73       861
algorithm       1       0.70      0.60      0.64      2649
caption         2       0.79      0.51      0.62       267
proof           3       0.75      0.76      0.75     99251
assumption      4       0.58      0.27      0.37      4640
definition      5       0.84      0.79      0.82    100324
problem         6       0.72      0.46      0.56      5570
remark          7       0.70      0.63      0.66     99476
other           8       0.87      0.90      0.89    572728

avg / total             0.83      0.84      0.83    885766

```

Discussion: 3 epochs seem sufficient. Performance only marginably better (1%) than single layer biLSTM

# TODO: Variants to try out
 * Embedding dropout of 0.5 instead of the biLSTM dropout
 * biLSTM + Attention = https://github.com/keras-team/keras/issues/4962#issuecomment-271934502 
 * 5-level deep biLSTM (what # of units in each? 100? and dropout? 0.2?)
 * biLSTMs seem to converge very fast, maybe 3 epochs max?