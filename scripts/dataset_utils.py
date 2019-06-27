# -*- coding: utf-8 -*-

import csv
from sklearn.utils import resample
import numpy as np
#from imblearn.over_sampling import RandomOverSampler
import random
from sklearn.model_selection import train_test_split

import sys
import string
if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

def text_to_word_sequence(text,
                          filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789',
                          lower=True, split=" "):
    try:
        if lower:
            text = text.lower()
        text = text.translate(maketrans(filters, split * len(filters)))
        seq = text.split(split)
        return [i for i in seq if i]
    except:
        print(text)


    
def load_data(data_path, header = True):
    data = open(data_path, 'r')
    data_reader = csv.reader(data, delimiter=',')
    if header:
        next(data_reader, None)  # skip the headers

    data = list(data_reader)
    return data

def save_data(data, data_path, header=None):
    data_f = open(data_path, 'w')
    data_writer = csv.writer(data_f, delimiter=',', quoting=csv.QUOTE_NONE,escapechar=' ')
    
    if header:
        data_writer.writerow(header)

    for d in data:
        data_writer.writerow(d)
        
    

def balance_dataset(dataset, dims, n_samples, method= None):
    
    if method == None or method == 'underampling':
        dataset = undersample_dataset(dataset, dims, n_samples)
        
    elif method == 'overampling':
        dataset = oversample_dataset(dataset, dims, n_samples)

    return dataset

def undersample_dataset(dataset, dims, n_samples):
    
    for dim in dims:
        col = np.array([d[dim] for d in dataset], dtype="int32")
        col_indexes = np.nonzero(col)
        if col_indexes[0].shape[0] < n_samples:
            continue
        random_indexes = np.random.choice([x for xs in col_indexes for x in xs], n_samples, replace=False)
        
        elements = [dataset[x] for x in random_indexes]
        col_indexes = np.where(col<1)
        col_indexes  = [x for xs in col_indexes for x in xs]
        dataset_tmp = [dataset[x] for x in col_indexes]

        dataset = elements + dataset_tmp

    return dataset

def oversample_dataset(dataset, dims, n_samples):
    
    for dim in dims:
        col = np.array([d[dim] for d in dataset], dtype="int")
        col_indexes = [x for xs in np.nonzero(col) for x in xs]
        if len(col_indexes) == 0:
            continue
        random_indexes = np.random.choice(col_indexes, n_samples, replace=True)
        
        elements = [dataset[x] for x in random_indexes]
        col_indexes = np.where(col<1)
        col_indexes  = [x for xs in col_indexes for x in xs]
        dataset_tmp = [dataset[x] for x in col_indexes]
        
        dataset = elements + dataset_tmp

    return dataset

def remove_cols(data, cols):
    for i, col in enumerate(cols):
        for row in data:
            del row[col-i]
    return data

def generate_radom_dataset():
    data = load_data("../../datasets/cado/cado_single_kn.csv")
    generated_data_set = []
    i=0
    while i < 1000:
        random_col = 0
        kn_start = 0
        kn_stop = 0        
        
        while True:
            random_col = np.random.randint(1, 11)
            if random_col != 5 and random_col != 9:
                break
        
        random_col_texts = [row[0] for row in data if int(row[random_col]) == 1]
        noninfo_texts = [row[0] for row in data if int(row[12]) == 1]
        
        random_info = np.random.choice(random_col_texts)
        random_noninfo1 = np.random.choice(noninfo_texts)
        random_noninfo2 = np.random.choice(noninfo_texts)
        
        if len(text_to_word_sequence(random_info)) < 20:
            continue
        
        
        infos = np.array([random_info, random_noninfo1, random_noninfo2])
        
        random_indicies = [1,0,2]
        #np.random.shuffle(random_indicies)
        
        if random_indicies.index(0) == 0:
            kn_start = 0
            kn_stop = len(text_to_word_sequence(infos[0]))
        elif random_indicies.index(0) == 1:
            kn_start = len(text_to_word_sequence(infos[random_indicies[0]]))
            kn_stop = kn_start + len(text_to_word_sequence(infos[0]))
        elif random_indicies.index(0) == 2:
            kn_start = len(text_to_word_sequence(infos[random_indicies[0]])) + len(text_to_word_sequence(infos[random_indicies[1]]))
            kn_stop = kn_start + len(text_to_word_sequence(infos[0]))

        paragraph = " ".join(infos[random_indicies])
        paragraph_len = len(text_to_word_sequence(paragraph))
        
        if(kn_start==kn_stop):
            continue
        
        assert(kn_start!=kn_stop)
        assert(kn_stop <= paragraph_len)
        
        len_p = len(text_to_word_sequence(paragraph))
        len_kn = kn_stop-kn_start
                
        if len_kn <=10 or len_kn/len_p > 0.3:
            continue
        
        generated_data_set.append([paragraph , kn_start, kn_stop, random_col])
        i+=1
        
    save_data(generated_data_set, '../../datasets/cado/artificial_single_kn_loci6.csv', ['document', 'start', 'end', 'knType'])

    
if __name__ == "__main__":
    
    head=['id,quoteId,elementId,projectId,elementName,,text,functionality,concept,directives,purpose,quality,control,structure,patterns,codeExamples,environment,reference,nonInformation,knCount,length,Three']
    #generate_radom_dataset()
    data = load_data('../../datasets/cado_new/train2.csv')
    data = [t for t in data if int(t[19]) ==1]
    XXX
    funcs = []
    non_infs = []
    new_data = []
    
    for d in data:
        if d[7] == '1':
            if int("".join(d[8:19]))==0:
                funcs.append(d)
            else:
                 new_data.append(d)
        elif d[18] == '1':
            if int("".join(d[7:18]))==0:
                non_infs.append(d)
            else:
                 new_data.append(d)
        else:
            new_data.append(d)
            
    prid_index = 3
    text_index = 6
    label_start_index = 7 

    random.shuffle(funcs)
    random.shuffle(non_infs)
    
    #new_data = new_data+funcs[1:10]+non_infs[1:10]
    
    new_data = data
    random.shuffle(new_data)
    
    #X_train, X_test = train_test_split(new_data, test_size=0.2, random_state=123)

    new_data=oversample_dataset(new_data, [8], 1000)
    new_data=oversample_dataset(new_data, [9], 1000)
    new_data=oversample_dataset(new_data, [10], 1000)
    new_data=oversample_dataset(new_data, [11], 1000)
    new_data=oversample_dataset(new_data, [12], 1000)
    new_data=oversample_dataset(new_data, [13], 1000)
    new_data=oversample_dataset(new_data, [14], 1000)
    new_data=oversample_dataset(new_data, [15], 1000)
    new_data=oversample_dataset(new_data, [16], 1000)
    new_data=oversample_dataset(new_data, [17], 1000)
    
    
    for k in range(12):
        print(len([x for x in data if x[label_start_index+k] == '1']))    
    
