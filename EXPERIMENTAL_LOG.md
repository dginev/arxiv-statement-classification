# LSTM(32), sgd optimizer, dense softmax run:
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
 411703/411703 [==============================] - 2702s 7ms/step - loss: 2.0567 - sparse_categorical_accuracy: 0.3320 - val_loss: 1.9705 - val_sparse_categorical_accuracy: 0.3538
 Evaluating model on test data...
 sparse_categorical_accuracy: 35.39%
 
```

# LSTM(64), adam, dense softmax run:

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

# LSTM(75), adam, strict labels, dense softmax run:

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

# LSTM(150), adam, strict labels, dense softmax run:
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

# LSTM(75), adam, strict labels, dense sigmoid run:
Not better, sticking to softmax
```
411703/411703 [==============================] - 1805s 4ms/step - loss: 1.3572 - sparse_categorical_accuracy: 0.5024
                                                            - val_loss: 1.2154 - val_sparse_categorical_accuracy: 0.5485

```
# LSTM(75)+LSTM(75), (adam, strict labels, dense softmax) run:

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

# LSTM(7*n_classes)+LSTM(7*n_classes)+Dense(3*n_classes), batch size = 100
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
