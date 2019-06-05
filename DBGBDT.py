# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:47:09 2019

@author: zhouhao1
"""
import numpy as np
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF


def negative_gradient(pred,label):
    numerator = 2 * label
    denominator = 1 + np.exp(2 * np.multiply(label,pred))
    return numerator / denominator

def expit(x):
    return np.exp(2*x)/(1+np.exp(2*x))


def initial(label):
         pred_initial = np.zeros_like(label, dtype = float)
         ybar = np.mean(label)
         pred_initial[:] = 0.5*np.log((1+ybar)/(1-ybar))
         return pred_initial

def loss_train(label,pred,sample_weight):   
    fra = -2.0 * np.multiply(label,pred)
    loss_ =  np.sum(np.multiply(sample_weight,np.logaddexp(0,fra)))
    return  loss_
      
def updata_terminal_region(tree,terminal_regions,leaf,pseudo_response,sample_weight):

    terminal_region = np.where(terminal_regions == leaf)
    pseudo_res_ = pseudo_response.take(terminal_region)
    sample_weight_ = sample_weight.take(terminal_region)
    numerator = np.sum(np.multiply(sample_weight_,pseudo_res_))
    denominator_ = np.multiply(np.abs(pseudo_res_),(2 - np.abs(pseudo_res_)))
    denominator = np.sum( np.multiply(sample_weight_,denominator_) )
        
    if np.abs(denominator) < 1e-150:
        tree.value[leaf,0,0] = 0.0
    else:
        tree.value[leaf,0,0] = numerator / denominator        
            
class gradientboostingclassifier:
    
    def __init__(self, 
                learning_rate = 0.1,
                n_estimators = 50, 
                bins = 10,
                momentum = 0,
                criterion = 'friedman_mse',
                splitter = 'best',
                max_depth=3,
                min_samples_split = 2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0,
                min_impurity_decrease=0,
                min_impurity_split=None,
                max_features=None,
                max_leaf_nodes=None,
                random_state=None,
                presort='auto'):
        
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.presort = presort
        self.bins = bins
        self.momentum = momentum
        self.trees = []
        self.y_pred = []
        self.proa = []
        self.label = []

        
    def fit(self,X,y):
        
        dtr = tree.DecisionTreeRegressor(
                criterion = self.criterion,
                splitter = self.splitter,
                max_depth = self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                min_impurity_split=self.min_impurity_split,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes)
        
        gd = GradientDensity(bins = self.bins,momentum = self.momentum)
        
        init = initial(y)  ##initial base learner
        self.label = y
        y_pred_array = np.zeros((X.shape[0],(self.n_estimators+1)), dtype = float)
        self.y_pred = np.asmatrix(y_pred_array)
        self.y_pred[:,0] = init
        sample_weight_0 = np.zeros((X.shape[0],self.n_estimators),dtype = float)
        self.sample_weight = np.asmatrix(sample_weight_0)
        ## boost
        for i in range(1,(self.n_estimators+1)):
            pseudo_response = negative_gradient(init,y)  ##negative gradient
            
            gnorm = np.abs(pseudo_response)
            sample_weight = gd.calc(gnorm)
            dtr = dtr.fit(X,pseudo_response)
            dtrtree_ = dtr.tree_
            terminal_regions = dtrtree_.apply(X)
            
            for leaf in np.where(dtrtree_.children_left == TREE_LEAF)[0]:               
                updata_terminal_region(dtrtree_,terminal_regions,leaf, pseudo_response,sample_weight)
            
            leaf_value = dtrtree_.value[:, 0, 0].take(terminal_regions)
            leaf_value = np.mat(leaf_value).T
            self.y_pred[:,i] = self.y_pred[:,(i-1)] + self.learning_rate * leaf_value
            self.sample_weight[:,(i-1)] = sample_weight
            self.trees.append(dtrtree_)
            init = self.y_pred[:,i]
        
    def loss(self):
        pred = self.y_pred[:,self.n_estimators]
        weight = self.sample_weight[:,(self.n_estimators-1)]
        loss_ = loss_train(self.label,pred,weight)
        return loss_       ## 一共有 self.n_estimators+1 棵树，第一棵是初始化，不需要查看loss
        
    def predict(self,X_test):
        X_test_rowcount = X_test.shape[0]
        trees = self.trees
        pre = self.y_pred[:,0][0:X_test_rowcount]  ## 预测初始�?
    
        for i in range(0,len(trees)):
            tree_ = trees[i]
            pre_ = tree_.predict(X_test)

            pre = pre + self.learning_rate * pre_ 
        pre = np.asmatrix(pre)   
        proa_array = np.zeros((X_test.shape[0],2))
        self.proa = np.asmatrix(proa_array)
        pro = expit(pre)
        self.proa[:,1] = pro
        self.proa[:,0] = 1- self.proa[:,1]
        #label = np.zeros_like(pro,dtype=np.int)
        #label[pro>=self.threshold] = 1
        #label[pro<self.threshold] = 0
        label = np.argmax(self.proa,axis = 1)
        
        return label
        
    def proa(self):
        return self.proa
     
class GradientDensity:
    def __init__(self, bins , momentum ):
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / (0.5 * bins) for x in range(bins+1)]
        
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def calc(self,gnorm):

        edges = self.edges
        mmt = self.momentum
        g = gnorm
        #N = g.shape[1]
        tot = 2.0 / self.bins
        weights = np.zeros_like(g)
        
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) 
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bin
                    weights[inds] =  self.acc_sum[i] 
                else:
                    weights[inds] =  num_in_bin 
        weights = tot / weights
        #weights = (tot*N) / weights
        #weights = weights / np.sum(weights)
        return weights 
 
'''        
# DB-GBDT test    
  
label = np.array([1,-1,1,1,1,-1,-1,1,1,-1,1,-1],dtype = int)
label = np.asmatrix(label)
label = label.T

X = np.array(np.random.rand(12,3),dtype = 'float32')     
X = np.asmatrix(X)
    
X_test = np.array(np.random.rand(5,3),dtype = 'float32')
X_test = np.asmatrix(X_test)


    
gbdt = gradientboostingclassifier(n_estimators = 100)
gbdt.fit(X,label)        
pre = gbdt.predict(X_test)
loss = gbdt.loss()
proa = gbdt.proa
a = gbdt.y_pred
b = gbdt.trees
c = gbdt.sample_weight
'''












