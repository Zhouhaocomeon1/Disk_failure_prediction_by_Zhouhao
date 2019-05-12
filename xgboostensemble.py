# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:55:08 2019

@author: zhouhao1
"""

import pandas as pd
import numpy as np
import random 
from xgboost.sklearn import XGBClassifier




def random_dataset(data,n=10,m=1000):  
    # n is the number of the dataset; 
    #and the m is the number of the disk in each dataset
    disk_name = list(set(data['serial_number']))
    
    dataset = []
    for i in range(n):
        random.seed(i)
        disk_ = random.sample(disk_name,m)
        dataset_ = data[data.serial_number.isin(disk_)]
        dataset.append(dataset_)
        
        #disk_name_ = []
        #for j in disk_name:
            #if j not in disk_:
                #disk_name_.append(j)
        #disk_name = disk_name_
    return dataset

class xgboost_ensemble:
    def __init__(self,
               n_estimators=500,
               learning_rate=0.10,
               max_depth=8,
               min_child_weight=1,
               gamma=0,
               colsample_bytree=0.8,
               subsample=1.0,
               scale_pos_weight=1,
               seed=10,
               objective='binary:logistic',
               gpu_id=0,
               max_bin=256,
               tree_method='gpu_hist'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.subsample = subsample
        self.seed = seed
        self.objective = objective
        self.gpu_id = gpu_id
        self.max_bin = max_bin
        self.tree_method = tree_method
        self.model = []
        
    def fit(self,X_good,X_bad):
        

        for i in range(len(X_good)):
            x_good = X_good[i]
            del x_good['serial_number']
            label_good = np.zeros((x_good.shape[0],),dtype=np.int)  
            label_bad = np.ones((X_bad.shape[0],),dtype=np.int)
            label = np.hstack((label_good,label_bad))
            train_data = pd.concat((x_good,X_bad),axis=0,ignore_index=True)
            
            xgboost = XGBClassifier(n_estimators =self.n_estimators,
                        learning_rate=self.learning_rate,
                        max_depth=self.max_depth, 
                        min_child_weight=self.min_child_weight, 
                        gamma=self.gamma, 
                        colsample_bytree=self.colsample_bytree,
                        subsample=self.subsample,
                        scale_pos_weight=self.scale_pos_weight,
                        seed=self.seed,
                        objective=self.objective,
                        gpu_id=self.gpu_id,
                        max_bin=self.max_bin,
                        tree_method=self.tree_method)
            xgboost.fit(train_data,label)
            self.model.append(xgboost)
            
    def predict(self,data):
        pre_result = np.zeros((data.shape[0],len(self.model)),dtype=np.int)
        flag = len(self.model) / 2
        for i in range(len(self.model)):
            model = self.model[i]
            pre = model.predict(data)
            
            pre_result[:,i] = pre
            
        vote_sum = np.sum(pre_result,axis=1)
        pre = np.zeros_like(vote_sum)
        pre[vote_sum >= flag] = 1
        
        return pre



        
    







   
        
        