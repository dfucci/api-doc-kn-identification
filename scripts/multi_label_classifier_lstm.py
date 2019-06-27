#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import LSTM, Embedding, Dense, Input, Flatten, Dropout, Activation
from keras.models import Sequential, Model
import keras.regularizers as regularizers

from sklearn.metrics import precision_recall_fscore_support
import time
import lstm_configs as lstm_configs
import dataset_utils as du
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import object_logger as ol

from keras import backend as K

def mcor(y_true, y_pred):
     #matthews_correlation
     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
     y_pred_neg = 1 - y_pred_pos
 
 
     y_pos = K.round(K.clip(y_true, 0, 1))
     y_neg = 1 - y_pos
 
 
     tp = K.sum(y_pos * y_pred_pos)
     tn = K.sum(y_neg * y_pred_neg)
 
 
     fp = K.sum(y_neg * y_pred_pos)
     fn = K.sum(y_pos * y_pred_neg)
 
 
     numerator = (tp * tn - fp * fn)
     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
 
 
     return numerator / (denominator + K.epsilon())




def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def tokenize_data(X):
    tokenizer = Tokenizer(num_words=lstm_configs.MAX_NB_WORDS)
    tokenizer.fit_on_texts(X)    
    
    return tokenizer

def get_glove_embeddings_index(path):
    embeddings_index = {}
    f = open(path)
    for line in f:
        try:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError:
            print("emb val err.")
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index))
    
    return embeddings_index

def get_data_embedings():
    #ToDo: select a word embeding method
    return None

if __name__ == '__main__':
    t = int(time.time())
    data_path = ''
    test_path = ''
    glove_path = ''
    result_path = '../results/'+str(t)+'/'
    
    train_embeddings = False


    np.random.seed(123)

    if len(sys.argv) > 1:
        glove_path = str(sys.argv[1])

    if len(sys.argv) > 2:
        lstm_configs.EMBEDDING_DIM = int(sys.argv[2])
        
    if len(sys.argv) > 3:
        data_path = str(sys.argv[3])

    if len(sys.argv) > 4:
        test_path = str(sys.argv[4])

    if len(sys.argv) > 5:
        result_path = str(sys.argv[5])

    if len(sys.argv) > 6:
        train_embeddings = bool(str(sys.argv[6]))

    
    head = [[0, "documentText"],
            [1, "functionality"],
            [2, "concept"],
            [3, "directives"],
            [4, "purpose"],
            [5, "quality"],
            [6, "control"],
            [7, "structure"],
            [8, "patterns"],
            [9, "codeExamples"],
            [10, "environment"],
            [11, "reference"],
            [12, "nonInformation"]]
    
    data = du.load_data(data_path)
    test = du.load_data(test_path)
    
    #prid_index = 1
    #text_index = 0
    #label_start_index = 1 

    prid_index = 3
    text_index = 6
    label_start_index = 7 
    X = [d[text_index] for d in data]
    labels = [d[label_start_index:label_start_index+12] for d in data ]
    pr_ids = np.array([d[prid_index] for d in data])
    
    
    prid_index = 3
    text_index = 6
    label_start_index = 7
    X_test = [d[text_index] for d in test]
    labels_test = [d[label_start_index:label_start_index+12] for d in test]
    pr_ids_test = np.array([d[prid_index] for d in test])
    
        
    
    Y = np.array(labels, dtype='int')
    y_test = np.array(labels_test, dtype='int')
    #Y = np.array(binary_labels, dtype='int')
    labels_index = {}  # dictionary mapping label name to numeric id
    
    test_index = len(X)
    
    X = X + X_test
    Y = np.vstack([Y , y_test])
    
    tokenizer = tokenize_data(X)
    word_index = tokenizer.word_index
    
    sequences = tokenizer.texts_to_sequences(X)    

    X = pad_sequences(sequences, maxlen=lstm_configs.MAX_SEQUENCE_LENGTH, 
                      padding="post", truncating="post", value=0)   

    all_tokens_num = sum([word[0] for word in enumerate(word_index)])
    print('%s documents.' % len(X))
    print('%s tokens.' % all_tokens_num)
    print('%s unique tokens.' % len(word_index))
    print('%s average tokens per document.' % int(sum([len(s) for s in sequences]) / len(X)))
    print('%s max tokens per document.' % int(max([len(s) for s in sequences])))
    print('%s min tokens per document.' % int(min([len(s) for s in sequences])))
    print('Shape of data tensor:', X.shape)
    print('Shape of label tensor:', Y.shape)

    print('Preparing embedding matrix.')
    
    embeddings_index = get_glove_embeddings_index(glove_path) 
    missing_words = {}
   
    # prepare embedding matrix
    num_words = min(lstm_configs.MAX_NB_WORDS, len(word_index)+1)
    embedding_matrix = np.zeros((num_words, lstm_configs.EMBEDDING_DIM))
        
    for word, i in word_index.items():
        if i >= lstm_configs.MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            missing_words[word]=word
            if train_embeddings:
                embedding_matrix[i] = np.random.rand(1,  lstm_configs.EMBEDDING_DIM)
            
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                lstm_configs.EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=lstm_configs.MAX_SEQUENCE_LENGTH,
                                trainable=train_embeddings, mask_zero=True)    

    print('Preparing taraining and validation matrix.')
    
    train_accuracies = []
    test_accuracies = []   
    test_results = []
    multi_label_results =[]

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    du.save_data([[data_path],[glove_path],[lstm_configs.MAX_NB_WORDS]],result_path+'/__meta')
    
    kf = KFold(n_splits=lstm_configs.K)
    k=0
        
    X_train = X[0:test_index , :]
    Y_train = Y[0:test_index , :]
    x_test = X[test_index:len(X), :]
    y_test = Y[test_index:len(Y), :]
    
    ######## init with co-ocurrance matrix
    #w=[np.zeros((Y.shape[1],Y.shape[1])), np.zeros((Y.shape[1]))]
    
    #cooccurrence_matrix = np.dot(Y_train.transpose(), Y_train)
    #cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
    #with np.errstate(divide='ignore', invalid='ignore'):
    #    cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))
    
    #w[0] = cooccurrence_matrix_percentage
    
    
    pr_ids_train = pr_ids
    test_data_buf=[]
    for train_index, val_index in kf.split(X_train):
        # split test set        
        x_train = X_train[train_index]
        y_train = Y_train[train_index]
        x_val = X_train[val_index]
        y_val = Y_train[val_index]

        

        print('Build model...')

        model = Sequential()
        model.add(embedding_layer)
        
        model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
        #model.add(Dense(256, activation='relu')) #dropout=0.2
        model.add(Dense(128, activation='relu')) #dropout=0.2
        model.add(Dense(64, activation='relu')) #dropout=0.2
        #model.add(Dense(32, activation='relu')) #dropout=0.2

        model.add(Dense(Y.shape[1], activation=lstm_configs.ACTIVATION_M)) # softmax
        #model.add(Dense(Y.shape[1], activation=lstm_configs.ACTIVATION_M, weights=w)) # softmax

        # try using different optimizers and different optimizer configs
        
        model.compile(loss=lstm_configs.LOSS_M, #lstm_configs.LOSS_M, # categorical_crossentropy
                      optimizer= lstm_configs.OPTIMIZER_M, # rmsprop
                      metrics=['accuracy', f1])
        
        print('Train...')
        
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')]
    
        batch_size = 32 
        
        hist = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=lstm_configs.EPOCHES,
                  validation_data=(x_val, y_val), callbacks=callbacks)

    
        ol.dump_obj_json(result_path+'itr'+str(k), hist.history)
        
        test_preds_pr = model.predict(x_test)
        test_preds = np.array(test_preds_pr)
        test_preds[test_preds>0.5] = 1
        test_preds[test_preds<=0.5] = 0
        
        test_preds_pr = test_preds_pr.astype('str')

        y_test = y_test.astype('str')
        test_preds = np.array([[int(float(p2)) for p2 in p1] for p1 in test_preds])
        test_preds = test_preds.astype('str')

        test_prediction_results  = []
        for i in range(len(test_preds)):
            test_prediction_results.append([",".join(test_preds[i])])
        
        test_prediction_results_pr  = []
        for i in range(len(test_preds_pr)):
            test_prediction_results_pr.append([",".join(test_preds_pr[i])])
        
        du.save_data(test_prediction_results,result_path+'p'+str(k),header=lstm_configs.head_short)          

        du.save_data(test_prediction_results_pr,result_path+'pr'+str(k),header=lstm_configs.head_short)          
        

        if not os.path.exists(result_path+'models/'):
            os.makedirs(result_path+'models/')
        model.save(result_path+'models/model'+str(k)+'.h5')
        
        if not os.path.exists(result_path+'train_sets/'):
            os.makedirs(result_path+'train_sets/')
        ol.dump_obj_json(result_path+'train_sets/t'+str(k), {'train':train_index.tolist(), 'val':val_index.tolist()})
        

        k += 1

    
    du.save_data([[word] for word in missing_words.keys()], result_path+'missing_words.csv')
    du.save_data([[i] for i in pr_ids_test.astype(str)],result_path+'test_pr_ids.csv')          
    
        
