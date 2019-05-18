# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:23:29 2019

@author: zhouhao1
"""
import pandas as pd
import numpy as np
import collections
from sklearn.ensemble import RandomForestClassifier as rf
import random
import os
#============================dataintefration===================================
def dataintegration(file_read_path,version):
    filepath = file_read_path
    os.chdir(filepath) 
    L=[]
    for files in os.walk(filepath):
        for file in files:
            L.append(file)       
    filename = L[2]

    data = pd.DataFrame()
    for i in filename:
        data0 = pd.read_csv(i)
        data0 = data0[data0.model==version]
        data = data.append(data0,ignore_index=True)
        data = data.dropna(axis = 1,how = 'all')
        print(i)
        #print(data0.shape)
        #print(data.shape)
        del data0,i
    print('===================dataintegration_result=========================')
    print('shape: (%d,%d)' % data.shape)
    print('capacity_bytes')
    print(set(data.capacity_bytes))
    print('total_count: %d' % len(set(data.serial_number)))
    print('bad_count: %d' % len(set(data.serial_number[data.failure == 1])))
    print('finish')

    data.dropna(axis = 0,how = 'any',inplace=True)
    print('after_delete')
    print('shape: (%d,%d)' % data.shape)
    print('capacity_bytes')
    print(set(data.capacity_bytes))
    print('total_count: %d' % len(set(data.serial_number)))
    print('bad_count: %d' % len(set(data.serial_number[data.failure == 1])))

    del  data['model'] , data['capacity_bytes']

    failurename = list(set(data.serial_number[data.failure == 1]))
    print('bad_count:')
    print(len(failurename))
    
    data.sort_values(by = ['serial_number','date'],ascending = True,inplace=True)
    
    #data.to_csv('ST4000DM000.csv', index = False)
    bad = data[data.serial_number.isin(failurename)]

    goodname = []
    diskname = list(set(data.serial_number))
    for i in diskname:
        if i not in failurename:
            goodname.append(i)

    
    good = data[data.serial_number.isin(goodname)]
    good = good.sort_values(by = ['serial_number','date'],ascending = True)
    bad = bad.sort_values(by = ['serial_number','date'],ascending = True)

    print('good.shape (%d,%d):' % good.shape)
    print('bad.shape (%d,%d):' % bad.shape)

    print('good_count')
    print(len(set(good['serial_number'])))
    print('bad_count')
    print(len(set(bad['serial_number'])))
    print('==================================================================')

    #good.to_csv('good.csv', index = False)
    #bad.to_csv('bad.csv', index = False)
    
    return good, bad
#==================================select_good_month===========================
def select_good_month(data,month):
    Date = data['date']
    date = []
    for i in Date:
        a = i[5:7]
        a = int(a)
        date.append(a)
    date = np.array(date)
    inde = np.where(date == month)
    inde = np.array(inde).ravel()
    good = data.iloc[inde,:]
    return good

#================================bad_label_anormal_detect======================
def anomaly_detect(df):
    name = df['serial_number']
    failure = df['failure']
    name = list(name)
    failure = list(failure)

    i = 0
    start = 0
    normal = []
    while i < (len(name)-1) :
        
        if name[i] != name[i+1]:
            end = i
            
            disk_con = failure[start:(end+1)]
            disk_con_num = 0
            for j in disk_con[:-1]:
                if j == 0:
                    disk_con_num += 1
            if disk_con_num != (len(disk_con)-1):
                normal.append(name[end])       

            start = end+1
        if i == (len(name)-2):
            end = i+1
            disk_con = failure[start:(end+1)]
            disk_con_num = 0
            for j in disk_con[:-1]:
                if j == 0:
                    disk_con_num += 1
            if disk_con_num != (len(disk_con)-1):
                normal.append(name[end])             
        i += 1
    return normal


def bad_label_anormaly_detect(data):
    disk_name = list(set(data['serial_number']))
    anormal_disk = anomaly_detect(data)
    anormal = data[data.serial_number.isin(anormal_disk)]
    normal = []
    for i in disk_name:
        if i not in anormal_disk:
            normal.append(i)

    bad_normal = data[data.serial_number.isin(normal)]
    
    print('===============bad_label_anormaly_detect_result===================')
    print('len(disk_name)%d'% len(disk_name))
    print('len(anormal_disk) %d' % len(anormal_disk))
    print('len(normal)%d' % len(normal))
    print('==================================================================')
    return anormal, bad_normal

#===============================data_process===================================
def data_des(data):  
    list_diskmodel = list(data['serial_number'])
    a = collections.Counter(list_diskmodel)
    b = pd.DataFrame(pd.Series(a),columns = ['count'])
    b = b.reset_index().rename(columns = {'index':'serial_number'})
    return b


def data_cut_out(data,
                 disk_type,
                 time_len):

    if disk_type == 'bad':
        datapro_bad = data_des(data)
        disk_count_bad =  np.array(datapro_bad['count'], dtype = int)
        timewindow = time_len
        delete_inde = np.argwhere(disk_count_bad<timewindow).ravel()
        
        print('=============data_cut_out_result==============================')
        print('before_cut_out')
        print('bad_shape: (%d,%d)' % data.shape)
        print('bad_disk_count: %d' % len(set(data.serial_number)))
        print('delete_count: %d' % len(delete_inde))
        print('==============================================================')
        
        serial_number_lessthan_tw = datapro_bad['serial_number'].iloc[delete_inde]
        serial_number_lessthan_tw = list(serial_number_lessthan_tw)
        serial_number_morethan_tw = []
        bad_name = list(datapro_bad['serial_number'])
        for i in bad_name:
            if i not in serial_number_lessthan_tw:
                serial_number_morethan_tw.append(i)
            del i

        data = data[data.serial_number.isin(serial_number_morethan_tw)]
        datapro_bad = data_des(data)
        disk_count_bad =  np.array(datapro_bad['count'], dtype = int)
        disk_cumu = np.cumsum(disk_count_bad)
        
        data_inde = []
        data_inde = np.array(data_inde,dtype = int)

        for i in np.arange(len(disk_count_bad)):
            data_ = np.arange((disk_cumu[i] - timewindow),disk_cumu[i])
            data_inde = np.concatenate((data_inde,data_),axis = 0)
            del data_,i
            
        bad = data.iloc[data_inde,:]
        return bad
    

    if disk_type == 'good':
        timewindow = time_len
        datapro_good = data_des(data)
        disk_count_good = np.array(datapro_good['count'], dtype = int)
        delete_inde = np.argwhere(disk_count_good<timewindow).ravel()
        
        print('========================data_cut_out_result===================')
        print('before_delete')
        print('good.shape: (%d,%d)' % data.shape)
        print('good_count: %d' % len(set(data.serial_number)))
        print('delete_count: %d' % len(delete_inde))
        print('==============================================================')
        
        serial_number_lessthan_tw = datapro_good['serial_number'].iloc[delete_inde]
        serial_number_lessthan_tw = list(serial_number_lessthan_tw)
        serial_number_morethan_tw = []
        good_name = list(datapro_good['serial_number'])
        for i in good_name:
            if i not in serial_number_lessthan_tw:
                serial_number_morethan_tw.append(i)
            del i  
        data = data[data.serial_number.isin(serial_number_morethan_tw)]

        datapro_good = data_des(data)
        disk_count_good =  np.array(datapro_good['count'], dtype = int)
        disk_cumu = np.cumsum(disk_count_good)
        
        data_inde = []
        data_inde = np.array(data_inde,dtype = int)

        for i in np.arange(len(disk_count_good)):
            data_ = np.arange((disk_cumu[i] - timewindow),disk_cumu[i])
            data_inde = np.concatenate((data_inde,data_),axis = 0)
            del data_,i
            
        good = data.iloc[data_inde,:]
        return good
#==============================================================================
        
#===============================feature_select=================================


def feature_select(good,bad,num,advance = True):
    
    train = pd.concat([good,bad],axis = 0,ignore_index = True)
    good_label = np.zeros((good.shape[0],1),dtype=np.int)
    bad_label = np.ones((bad.shape[0],1),dtype=np.int)
    train_label = np.vstack((good_label,bad_label))
    train_label = train_label.ravel()
    train_label = pd.Series(train_label)
    train.drop(columns = ['serial_number'],inplace = True)
    
## feature select
    RF = rf(n_estimators=100,max_depth=5,max_features=None,random_state=10,n_jobs=5)
    RF.fit(train,train_label)
    feature_importance = RF.feature_importances_
    feature_importance = np.mat(feature_importance).T
    feature = list(train.columns)
    feature_score = pd.DataFrame(feature_importance)
    feature_score['feature'] = feature
    feature_score.rename(columns = {0:'importance'},inplace = True)
    feature_score = feature_score.sort_values(by = 'importance',ascending = False)

## select num feature
    Feature = feature_score['feature']
    type_ = []
    for i in Feature:
        a = i[-3:]
        type_.append(a)
        
    type_ = np.array(type_)
    inde = np.argwhere(type_ == 'raw').ravel()
    feature_score_raw = feature_score.iloc[inde,:]
    
    feature_select = feature_score_raw.iloc[0:num,1]
    print('=============resutl_feature_select_advance========================')
    print(feature_select)
    print(feature_score_raw.iloc[0:num,:])
    print('==================================================================')
    feature_select = list(feature_select)
    if advance:
        feature_select.insert(0,'serial_number')
    return feature_select
#==============================================================================
    
#=============================feature_engineer=================================
def MA(ts,win):
    # eg ts = 1 2 3 4 5
    # index   0 1 2 3 4
    ts_MA = np.zeros_like(ts,dtype = np.float)
    ts_MA[0:win] = np.nan
    for j in range(len(ts)-win):
        mean = np.mean(ts[j:(j+win)])
        ts_MA[(j+win)] = mean
        
    return ts_MA
        
#ts = np.arange(6)
#tsMA = MA(ts,2)

def EWMA(ts,alpha):
    ts_EWMA = np.zeros_like(ts,dtype = np.float)
  
    for j in range(len(ts)):
        if j==0:
            ts_EWMA[j] = ts[j]
        else:
            ewma = alpha * ts[j] + (1-alpha) * ts_EWMA[(j-1)]
            ts_EWMA[j] = ewma
    
    return ts_EWMA

#ts = np.arange(1,6)
#tsEWMA = EWMA(ts,0.9)

def DIFF(ts,win):
    ts_DIFF = np.zeros_like(ts,dtype = np.float)
    ts_DIFF[0:win] = np.nan
    for j in range(len(ts)-win):
        ts_DIFF[j+win] = ts[j+win] - ts[j]
    return ts_DIFF

#ts = np.arange(1,6)
#tsDIFF = DIFF(ts,1)


def feature_ma(df,win):
    name = df['serial_number']
    name = list(name)
    feature = pd.DataFrame()
    for k in range(1,len(df.columns)):
        feature__ = df.iloc[:,k]
        feature__ = np.array(feature__)
        i = 0
        start = 0
        feature_ = np.zeros_like(feature__,dtype = np.float)
    
        while i < (len(name)-1) :
        
            if name[i] != name[i+1]:
                end = i
                ts = feature__[start:(end+1)]

                feature_[start:(end+1)] = MA(ts,win)
                start = end+1
            
            if i == (len(name)-2):
                end = i+1
                ts = feature__[start:(end+1)]
                feature_[start:(end+1)] = MA(ts,win)
            
            i += 1
        feature_ = pd.Series(feature_)
        feature = pd.concat([feature,feature_],axis=1)
    return feature

def feature_ewma(df,alpha):
    name = df['serial_number']
    name = list(name)
    feature = pd.DataFrame()
    for k in range(1,len(df.columns)):
        feature__ = df.iloc[:,k]
        feature__ = np.array(feature__)
        i = 0
        start = 0
        feature_ = np.zeros_like(feature__,dtype = np.float)
    
        while i < (len(name)-1) :
        
            if name[i] != name[i+1]:
                end = i
                ts = feature__[start:(end+1)]

                feature_[start:(end+1)] = EWMA(ts,alpha)
                start = end+1
            
            if i == (len(name)-2):
                end = i+1
                ts = feature__[start:(end+1)]
                feature_[start:(end+1)] = EWMA(ts,alpha)
            
            i += 1
        feature_ = pd.Series(feature_)
        feature = pd.concat([feature,feature_],axis=1)
    return feature

def feature_diff(df,win):
    name = df['serial_number']
    name = list(name)
    feature = pd.DataFrame()
    for k in range(1,len(df.columns)):
        feature__ = df.iloc[:,k]
        feature__ = np.array(feature__)
        i = 0
        start = 0
        feature_ = np.zeros_like(feature__,dtype = np.float)
    
        while i < (len(name)-1) :
        
            if name[i] != name[i+1]:
                end = i
                ts = feature__[start:(end+1)]

                feature_[start:(end+1)] = DIFF(ts,win)
                start = end+1
            
            if i == (len(name)-2):
                end = i+1
                ts = feature__[start:(end+1)]
                feature_[start:(end+1)] = DIFF(ts,win)
            
            i += 1
        feature_ = pd.Series(feature_)
        feature = pd.concat([feature,feature_],axis=1)
    return feature


def feature_name(feature,process_type,param_num):
    # process_type param_num is string
    # process_type include ma, ewma, diff

    feature = list(feature)
    feature.remove('serial_number')
    for i in range(len(feature)):
        feature[i] = (feature[i]+'_'+process_type+'_'+param_num)
    return feature


def feature_process(good,bad):
    feature = good.columns
    feature = list(feature)
    
    
    ma_feature1 = feature_ma(bad,3)
    ma_feature2 = feature_ma(bad,5)

    ewma_feature1 = feature_ewma(bad,0.9)
    ewma_feature2 = feature_ewma(bad,0.7)
    ewma_feature3 = feature_ewma(bad,0.5)

    diff_feature1 = feature_diff(bad,1)
    diff_feature2 = feature_diff(bad,3)
    diff_feature3 = feature_diff(bad,5)

## change name   
    ma_feature1.columns = feature_name(feature,'ma','1')
    ma_feature2.columns = feature_name(feature,'ma','2')
 
    ewma_feature1.columns = feature_name(feature,'ewma','1')
    ewma_feature2.columns = feature_name(feature,'ewma','2')
    ewma_feature3.columns = feature_name(feature,'ewma','3')

    diff_feature1.columns = feature_name(feature,'diff','1')
    diff_feature2.columns = feature_name(feature,'diff','2')
    diff_feature3.columns = feature_name(feature,'diff','3')
   

    bad_ = pd.concat([bad,ma_feature1,ma_feature2,ewma_feature1,ewma_feature2,ewma_feature3,diff_feature1,diff_feature2,diff_feature3],
                    axis = 1)


    ma_feature1 = feature_ma(good,3)
    ma_feature2 = feature_ma(good,5)

    ewma_feature1 = feature_ewma(good,0.9)
    ewma_feature2 = feature_ewma(good,0.7)
    ewma_feature3 = feature_ewma(good,0.5)

    diff_feature1 = feature_diff(good,1)
    diff_feature2 = feature_diff(good,3)
    diff_feature3 = feature_diff(good,5)

## change name
    ma_feature1.columns = feature_name(feature,'ma','1')
    ma_feature2.columns = feature_name(feature,'ma','2')
 
    ewma_feature1.columns = feature_name(feature,'ewma','1')
    ewma_feature2.columns = feature_name(feature,'ewma','2')
    ewma_feature3.columns = feature_name(feature,'ewma','3')

    diff_feature1.columns = feature_name(feature,'diff','1')
    diff_feature2.columns = feature_name(feature,'diff','2')
    diff_feature3.columns = feature_name(feature,'diff','3')
 

    good_ = pd.concat([good,ma_feature1,ma_feature2,ewma_feature1,ewma_feature2,ewma_feature3,diff_feature1,diff_feature2,diff_feature3],
                     axis = 1)

    print('===================feature_engineering_result=====================')
    print('before')
    print('good.shape %d %d' %(good_.shape))
    print('bad.shape %d %d' %(bad_.shape))
    good_.dropna(axis=0,how='any',inplace=True)
    bad_.dropna(axis=0,how='any',inplace=True)
    print('after')
    print('good.shape %d %d' %(good_.shape))
    print('bad.shape %d %d' %(bad_.shape))
    print('==================================================================')
    
    return good_,bad_

#==============================================================================
    
#============================data_partion======================================

def exporttxt(filepath,data):
    with open(filepath,"w") as f:
        for i in data:
            f.write(i)
            f.write("\n")
            
def train_test_split(good,bad,perfen=0.7):
    
    ## bad
    datapro_bad = data_des(bad)
    bad_name = list(datapro_bad['serial_number'])
    random.seed(10)
    percen = 0.7

    N = np.rint(len(bad_name) * percen)
    N = N.astype(int)
    bad_train_name = random.sample(bad_name,N)
    bad_test_name = []

    for i in bad_name:
        if i not in bad_train_name:
            bad_test_name.append(i)
    
    badtrain = bad[bad.serial_number.isin(bad_train_name)]
    #bad_train = bad_train.sort_values(by = ['serial_number','date'],ascending = True)
    badtest = bad[bad.serial_number.isin(bad_test_name)]
    #bad_test = bad_test.sort_values(by = ['serial_number','date'],ascending = True)
    
    print('==========================bad_partion_result======================')
    print('bad')
    print('bad_train_count: %d' % len(bad_train_name))
    print('bad_test_count: %d' % len(bad_test_name))
    print('bad_train_shape: (%d,%d)' % badtrain.shape)
    print('bad_test_shape: (%d,%d)' % badtest.shape)
    
## good
    datapro_good = data_des(good)
    disk_count_good = np.array(datapro_good['count'], dtype = int)

    percen = 0.7
    disk_goodtrain_count = np.rint(disk_count_good *  percen)
    disk_goodtest_count = disk_count_good - disk_goodtrain_count

    disk_goodtrain_count = disk_goodtrain_count.astype(np.int)
    disk_goodtest_count = disk_goodtest_count.astype(np.int)
    disk_cumu = np.cumsum(disk_count_good)

    goodtrain = []
    goodtest = []
    goodtrain = np.array(goodtrain,dtype = int)
    goodtest = np.array(goodtest,dtype = int)
    start = 0
    for i in np.arange(len(disk_count_good)):
        train_ = np.arange(start,(start+disk_goodtrain_count[i]))
        test_ = np.arange((start+disk_goodtrain_count[i]),disk_cumu[i])
        goodtrain = np.concatenate((goodtrain,train_),axis = 0)
        goodtest = np.concatenate((goodtest,test_),axis = 0)
        start += disk_count_good[i]   
        del train_,test_,i
    
    goodtrain = good.iloc[goodtrain,:]
    goodtest = good.iloc[goodtest,:]
  

    #goodtrain = goodtrain.sort_values(by = ['serial_number','date'],ascending = True)
    #goodtest = goodtest.sort_values(by = ['serial_number','date'],ascending = True)

    print('good')
    print('goodtrain goodtest_count: %d' % len(set(goodtrain.serial_number)))
    print('goodtrain.shape: (%d,%d)' % goodtrain.shape)
    print('goodtest.shape: (%d,%d)' % goodtest.shape)
    print('==================================================================')
    
    goodtestname = goodtest['serial_number']
    badtestname = badtest['serial_number']
    
    path_current = os.getcwd()
    exporttxt(path_current+'/badtestname.txt',badtestname)
    exporttxt(path_current+'/goodtestname.txt',goodtestname)
    
    #badtrain.to_csv('badtrain.csv',index = False)
    #badtest.to_csv('badtest.csv',index = False)
    #goodtrain.to_csv('goodtrain.csv',index = False)
    #goodtest.to_csv('goodtest.csv',index = False)
    
    
    return goodtrain, goodtest, badtrain, badtest    
