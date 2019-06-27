#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import dataset_utils as du

def dump_obj_json(path, obj):
    with open(path, 'w') as file:
         file.write(json.dumps(obj))    
         
def load_obj_json(path):
    with open(path, 'r') as file:
         return json.loads(file.readline())    

def padd_array(arr):
    max_len = max([len(a) for a in arr])    
    arr = [a + ([a[-1]]*(max_len-len(a))) for a in arr]         
    return arr

def get_mean_std(arr):
    arr = np.array(arr)
    m = np.mean(arr)
    m = np.nanmean(arr, axis=0)
    s = np.nanstd(arr, axis=0)
    np.nan_to_num(s)
    np.nan_to_num(m)    
    return m, s
if __name__ == "__main__":
    f1=[]
    acc=[]
    loss = []
    for i in range(10):
        m = load_obj_json('../../results/itr'+str(i))
        f1.append(m['f1'])
        acc.append(m['acc'])
        loss.append(m['loss'])

    f1 = padd_array(f1)    
    f_m, f_s = get_mean_std(f1)    
    acc = padd_array(acc)        
    a_m, a_s = get_mean_std(acc)    
    loss = padd_array(loss)        
    l_m, l_s = get_mean_std(loss)    

    r = np.zeros((3*len(f_m),4))
    r = r.astype('str')
    r[:,0] = list(range(1,len(f_m)+1))*3
    r[:,1] = f_m.tolist() + a_m.tolist() + l_m.tolist()
    r[:,2] = f_s.tolist() + a_s.tolist() + l_s.tolist()
    r[:,3] = ['f1-score']*len(f_m) + ['accuracy']*len(a_m) + ['loss']*len(l_m)
    
    head=['epoch', 'mean', 'std', 'metric']
    
    du.save_data(r, '../../results/lstm1_medical_train_sum', head)