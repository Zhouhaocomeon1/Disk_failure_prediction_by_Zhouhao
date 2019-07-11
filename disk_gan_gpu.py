# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:54:05 2019

@author: zhouhao1
"""

import torch
import torch.nn as nn
#import torch.utils.data as Data
from torch.autograd import Variable
import os
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='disk_GAN')
parser.add_argument('--path', type=str, default='/data/zh/data_raw/HGST_BLE_2018/data')
parser.add_argument('--sequence_length', type=int, default=30,
                    help='the length of the sequence')
parser.add_argument('--G_input_size', type=int, default=10,
                    help='the dimension of the data in Generator')
parser.add_argument('--G_hidden_size', type=int, default=128,
                    help='the size of hidden in  Generator NN')
parser.add_argument('--G_output_size', type=int, default=48,
                    help='the output_size in G')
parser.add_argument('--G_num_layers', type=int, default=48,
                    help='the number of the hidden layers in Generotor NN')
parser.add_argument('--D_input_size', type=int, default=48,
                    help='the dimension of the data in Discrimanator')
parser.add_argument('--D_hidden_size', type=int, default=128,
                    help='the size of hidden in  Discrimanator NN')
parser.add_argument('--D_output_size', type=int, default=1,
                    help='the output_size in D')
parser.add_argument('--D_num_layers', type=int, default=1,
                    help='the number of the hidden layers in Discrimanator NN')

#parser.add_argument('--batch_size', type=int, default=30)
#parser.add_argument('--batch_disk_size', type=int, default=277)
#parser.add_argument('--num_epochs', type=int, default=100,
#                    help='upper epoch limit')
parser.add_argument('--G_learning_rate',type=float, default=0.01,
                    help='G_learning_rate')
parser.add_argument('--D_learning_rate',type=float, default=0.01,
                    help='D_learning_rate')
parser.add_argument('--seed', type=int, default=10,
                    help='seed')
parser.add_argument('--gpu_id',type=str,default="2",help='gpu id')
args = parser.parse_args()

# Hyper Parameters
sequence_length = args.sequence_length
  
G_input_size = args.G_input_size   #10
G_hidden_size = args.G_hidden_size #128
G_output_size = args.G_output_size  #48
G_num_layers = args.G_num_layers   #2

D_input_size = args.D_input_size    #48
D_hidden_size = args.D_hidden_size  #128
D_output_size = args.D_output_size  #1
D_num_layers = args.D_num_layers    #2

G_learning_rate = args.G_learning_rate
D_learning_rate = args.D_learning_rate

path = args.path
#batch_size = args.batch_size
#batch_disk_size = args.batch_disk_size
#num_epochs = args.num_epochs

os.chdir(path)
print('Hyper parameter')
#Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
a = torch.zeros(5).cuda()
print('Set GPU')
#Set random seed
torch.cuda.manual_seed(args.seed)
#torch.manual_seed(args.seed)
print('Set random seed')

# Define Generator
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        # Forward propagate RNN
        out, h_n = self.gru(x, h0)  
        #print(out.shape)        
        # Decode hidden state of last time step
        out = self.fc(out)  
        #print(out.shape)
        return out
# Define Discriminator    
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        # Forward propagate RNN
        out, h_n = self.gru(x, h0)  
        #print(out.shape)        
        # Decode hidden state of last time step
        out = self.fc(out)
        output = self.sigmoid(out)
        output = output.view((x.size(0)*x.size(1)))
        #print(out.shape)
        return output  
    
# the input of Generator
def noise(batch,sample_num,feature_num):
    noise_data = np.random.rand(batch,sample_num,feature_num)
    noise_data = np.around(noise_data,decimals=4)
    noise_data = torch.from_numpy(noise_data).float()
    noise_data = Variable(noise_data)
    return noise_data
    

def load_disk_data(size_per):
    #size_per = 8
    #8 * 1273 = 10184
    goodtrain = pd.read_csv('goodtrain.csv')
    train_data = torch.FloatTensor(goodtrain.values)
    #len = train_data.size(0)
    #batch_num = int(len / size_per)
    train_data_ = train_data.view(size_per,-1,D_input_size)
    train_data_ = list(train_data_)
    data_iter = iter(train_data_)
    return data_iter

G = Generator(G_input_size,G_hidden_size,G_num_layers,G_output_size).cuda()
D = Discriminator(D_input_size,D_hidden_size,D_num_layers,D_output_size).cuda()

opt_D = torch.optim.Adam(D.parameters(), lr = D_learning_rate)
opt_G = torch.optim.Adam(G.parameters(), lr = G_learning_rate)

for step in range(1000):
    sample_noise = noise(1000,sequence_length,G_input_size).cuda()
    sample_data = G(sample_noise)
    real_disk_data = load_disk_data(8)
    # train Discriminator
    for epoch in range(50):
        prob_fake_on_D = D(sample_data)
        for real_data in real_disk_data:
          real_data_gpu = Variable(real_data.view(-1,sequence_length,D_input_size)).cuda()
          prob_real_on_D = D(real_data_gpu)
          # the D_loss is from two part: one is from fake_data on D, the other is from real data on D
          D_loss_on_fake = -torch.mean(torch.log( 1.001 - prob_fake_on_D )) # loss from fake data
          D_loss_on_real = torch.mean(torch.log(prob_real_on_D)) # loss from real data
          D_loss = - D_loss_on_fake + D_loss_on_real # total loss on D
          #print(D_loss.shape)
          opt_D.zero_grad()
          D_loss.backward(retain_graph=True)
          opt_D.step()
    print('[step / total_step: %d//%d] D_loss is : %.4f' %(step,1000,D_loss.cpu().data))
    #train Generator
    prob_g_samble_on_G_after_train_D = D(sample_data)
    #print(torch.sum(prob_g_samble_on_G_after_train_D))
    G_loss = torch.mean(torch.log(1.001 - prob_g_samble_on_G_after_train_D)) # the loss on G
    opt_G.zero_grad()
    G_loss.backward(retain_graph=True)
    opt_G.step()
    print('[step / total_step %d//%d] G_loss is : %.4f' %(step,1000,G_loss.cpu().data))

torch.save(G.cpu().state_dict(),'gan_g_gpu.pkl')
torch.save(D.cpu().state_dict(),'gan_d_gpu.pkl')
        
    
    

