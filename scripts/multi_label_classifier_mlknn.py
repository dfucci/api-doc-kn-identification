#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dataset_utils as du
from skmultilearn.adapt import MLkNN
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import sys

MAX_NB_WORDS =20000

def tokenize_data(X):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X)       
    return tokenizer

def get_cado_predictions():
    data_path = '../../datasets/cado/train.csv'
    test_path = '../../datasets/cado/test.csv'
    
    data = du.load_data(data_path)
    test = du.load_data(test_path)
    
    text_index = 6
    label_start_index = 7 
    X = [d[text_index] for d in data]
    labels = [d[label_start_index:label_start_index+12] for d in data ]
    
    
    X_test = [d[text_index] for d in test]
    labels_test = [d[label_start_index:label_start_index+12] for d in test]
    
        
    
    Y = np.array(labels, dtype='int')
    y_test = np.array(labels_test, dtype='int')
    #Y = np.array(binary_labels, dtype='int')
    
    test_index = len(X)
    
    X = X + X_test
    Y = np.vstack([Y , y_test])
    
    tokenizer = tokenize_data(X)
    word_index = tokenizer.word_index
    
    sequences = tokenizer.texts_to_sequences(X)    

    X = pad_sequences(sequences, maxlen=700, 
                      padding="post", truncating="post", value=0)   

    num_words = min(MAX_NB_WORDS, len(word_index)+1)
    embedding_matrix = np.zeros((num_words, 1))

    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_matrix[i] = 1
        
    X_train = X[0:test_index , :]
    Y_train = Y[0:test_index , :]
    x_test = X[test_index:len(X), :]
    y_test = Y[test_index:len(Y), :]
    
    classifier = MLkNN()
    classifier.fit(X_train,Y_train)
    predictions = classifier.predict(x_test)
    scores = classifier.predict_proba(x_test)
    y_pred= predictions.toarray()
    y_score= scores.toarray()

    return y_pred, y_score
    
if __name__ == "__main__":
    
    p, pr = get_cado_predictions()
    
