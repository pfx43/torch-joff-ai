# -*- coding: utf-8 -*-
"""
画可视化图

@author: lenovo
"""

import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from math import sqrt
device = torch.device("cuda:0")	# 使用gpu训练

# 文件读取及预处理
def get_Data(data_path1,data_path2):
    
    # 训练数据 存放在data_path1
    data1=pd.read_excel(data_path1, 1)
    data1=data1.iloc[:,:]
    data1=data1.values
    # 测试数据 存放在data_path2
    data2=pd.read_excel(data_path2, 1)
    data2=data2.iloc[:,:]
    data2=data2.values
    
    # 归一化
    #data=MinMaxScaler().fit_transform(data) # 对数据进行归一化等处理
    # 标准化
    #object = StandardScaler()
    #data1 = object.fit_transform(data1)
    #data2 = object.transform(data2)
    
    return data1,data2

# 选取x u y
def split_windows(data,seq_length,input_size,u_num,y_num):
    
    # 选取小块中的堆叠u
    data_spilo_u = data[:,:u_num]
    df_u = pd.DataFrame(data_spilo_u)
    x_rows = [df_u[k:k+seq_length]
                    for k in range(0, len(df_u)-seq_length+1, 1)]
    x_rows_values=[x_rows[k].values
                    for k in range(0, len(x_rows), 1)]
    x=np.zeros((len(x_rows),seq_length * u_num))# 矩阵存储空间
    for j in range(0,len(x_rows),1):
        m=x_rows_values[j]
        x[j]=m.reshape(1,seq_length * u_num)
    x=x[:-2,:]
    # 选取小块中的堆叠y
    data_spilo_y = data[:,-y_num:]
    df_y = pd.DataFrame(data_spilo_y)
    y_rows = [df_y[k:k+seq_length]
                    for k in range(0, len(df_y)-seq_length+1, 1)]
    y_rows_values=[y_rows[k].values
                    for k in range(0, len(y_rows), 1)]
    y=np.zeros((len(y_rows),seq_length * y_num))# 矩阵存储空间
    for q in range(0,len(y_rows),1):
        m=y_rows_values[q]
        y[q]=m.reshape(1,seq_length * y_num)
    y=y[:-2,:]
    y=np.array(y)
    #选取u
    u=[]
    for q in range(len(data)-seq_length-1): # range的范围需要减去时间步长和1
        _u=data[q+seq_length,:u_num] # 选取uy
        u.append(_u)
    u=np.array(u)
    # 选取标签y1
    y1=[]
    for e in range(len(data)-seq_length-1): # range的范围需要减去时间步长和1
        _y=data[e+seq_length+1,-y_num:] # 选取y
        y1.append(_y)
    y1=np.array(y1)
    return x,y,u,y1

class Tanhshrink(nn.Module):
    def __init__(self):
        super(Tanhshrink, self).__init__()

    def forward(self, x):
        return x - torch.tanh(x)

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        softplus_approx = torch.log1p(torch.exp(x))
        mish = x * torch.tanh(softplus_approx)
        return mish

class Gaussian(nn.Module):
    def __init__(self):
        super(Gaussian, self).__init__()

    def forward(self, x):
        gaussian = torch.exp(-torch.pow(x, 2))
        return gaussian

class FCNNModel(torch.nn.Module):
    def __init__(self,input_num,hidden_num1,hidden_num2,output_num):
        super().__init__()
        self.input_num=input_num
        self.hidden_num1=hidden_num1
        self.hidden_num2=hidden_num2
        self.output_num=output_num
        self.layer1 = nn.Sequential(nn.Linear(self.input_num, self.hidden_num1), nn.SiLU())
        self.layer2 = nn.Sequential(nn.Linear(self.hidden_num1, self.hidden_num2), nn.SiLU())
        self.layer3 = nn.Sequential(nn.Linear(self.hidden_num2, self.output_num))
    def forward(self,input):
        input = self.layer1(input)
        input = self.layer2(input)
        output = self.layer3(input).to(device)
        return output

def get_max(A, L, C, x_num, y_num, seq_length):
    I = torch.eye(x_num).to(device)  # 单位矩阵
    A_expend = A - torch.mm(L, C)
    A_expend_max_list = [I]  # 将单位矩阵加入A_max_list中
    L_max_list = [L.T]  # 将L.T矩阵加入Lmax_list中
    for i in range(seq_length-1):
        A_expend_max_list.append(torch.mm(A_expend_max_list[i], A_expend.T))  # 递归计算矩阵A_的i次方
        L_max_list.append(torch.mm(L_max_list[i], A_expend.T))
    A_expend_max = torch.cat(A_expend_max_list, dim=0)  # 所有矩阵上下拼接 30*3
    L_max = torch.cat(L_max_list, dim=0) # 20*3
    return A_expend_max,L_max

# 参数设置
path1='../data1024.xlsx'
path2='../data1024.xlsx'
seq_length=10 # 时间步长40 10
batch_size=16 # 16 32
u_num=1
y_num=1
x_num=4
input_size=u_num+y_num
input_num=u_num
hidden_num1=3
hidden_num2=3
output_num=x_num
C = np.array([[1, 0, 0, 0]])
C = torch.from_numpy(C).float().to(device)
# 读取模型参数
m_state_dict = torch.load('./trained_model/train01/model04.pt')
new_m = FCNNModel(input_num,hidden_num1,hidden_num2,output_num).to(device)
new_m.load_state_dict(m_state_dict)
A_data = torch.load('./trained_model/train01/trained_A04.pt')
L_data = torch.load('./trained_model/train01/trained_L04.pt')
A = nn.Parameter(A_data)
L = nn.Parameter(L_data)
A_expend_max,L_max=get_max(A, L, C, x_num, y_num, seq_length)
A0_data = torch.load('./trained_model/train01/initial_A04.pt')
L0_data = torch.load('./trained_model/train01/initial_L04.pt')
A0 = nn.Parameter(A0_data)
L0 = nn.Parameter(L0_data)

if __name__ == '__main__':
    
    data1,data2 = get_Data(path1, path2)
    # data1:训练数据Y1;  data2:测试数据Y2
    Y1_real=[]
    Y1_hat=[]
    Y2_real=[]
    Y2_hat=[]
    # 得到全部小训练集合并在一起的训练结果
    data1_spilo = data1[:,:]
    x_train,y_train,u_train,y1_train = split_windows(data1_spilo, seq_length, input_size, u_num, y_num)
    x_train,y_train,u_train,y1_train = Variable(torch.Tensor(np.array(x_train))),Variable(torch.Tensor(np.array(y_train))),Variable(torch.Tensor(np.array(u_train))),Variable(torch.Tensor(np.array(y1_train)))
    x_train,y_train,u_train,y1_train = x_train.to(device),y_train.to(device),u_train.to(device),y1_train.to(device)
    # 把x_train左右切割为小段，段的个数为seq_length
    train_chunks = torch.chunk(x_train, seq_length, 1)
    train_new_chunks = [new_m(train_chunk) for train_chunk in train_chunks]
    train_result = torch.cat(train_new_chunks, dim=1)
    train_state11 = torch.mm(train_result, A_expend_max)
    train_state12 = torch.mm(y_train, L_max)
    train_state1 = torch.add(train_state11,train_state12)
    train_state21 = torch.mm(train_state1, A.T)
    train_state22 = new_m(u_train)
    train_state2 = torch.add(train_state21,train_state22)
    train_yhat = torch.mm(train_state2, C.T)
    train_yhat_array = train_yhat.cpu().detach().numpy()
    train_yreal_array = y1_train.cpu().detach().numpy()
    
    data2_spilo = data2[:,:]
    x_test,y_test,u_test,y1_test = split_windows(data2_spilo, seq_length, input_size, u_num, y_num)
    x_test,y_test,u_test,y1_test = Variable(torch.Tensor(np.array(x_test))),Variable(torch.Tensor(np.array(y_test))),Variable(torch.Tensor(np.array(u_test))),Variable(torch.Tensor(np.array(y1_test)))
    x_test,y_test,u_test,y1_test = x_test.to(device),y_test.to(device),u_test.to(device),y1_test.to(device)
    # 把x_test左右切割为小段，段的个数为seq_length
    test_chunks = torch.chunk(x_test, seq_length, 1)
    test_new_chunks = [new_m(test_chunk) for test_chunk in test_chunks]
    test_result = torch.cat(test_new_chunks, dim=1)
    test_state11 = torch.mm(test_result, A_expend_max)
    test_state12 = torch.mm(y_test, L_max)
    test_state1 = torch.add(test_state11,test_state12)
    test_state21 = torch.mm(test_state1, A.T)
    test_state22 = new_m(u_test)
    test_state2 = torch.add(test_state21,test_state22)
    test_yhat = torch.mm(test_state2, C.T)
    test_yhat_array = test_yhat.cpu().detach().numpy()
    test_yreal_array = y1_test.cpu().detach().numpy()
    
    # 可视化
    # 第一张图 训练集y
    fig1 = plt.figure(dpi=1000)
    plt.title('train_y')
    plt.plot(train_yhat_array,'r')
    plt.plot(train_yreal_array)
    #plt.xlim(13500,14000)
    #plt.autoscale(axis='y')
    plt.show()

    # 第三张图 测试集y
    fig2 = plt.figure(dpi=1000)
    plt.title('test_y')
    plt.plot(test_yhat_array,'r')
    plt.plot(test_yreal_array)
    plt.xlim(15000,15500)
    plt.ylim(325,335)
    # plt.autoscale(axis='y')
    plt.show()
    
    print('-----------------------矩阵取值------------------------------')
    print("初始的矩阵A:", A0)
    print("训练后的矩阵A:", A)
    print("初始的矩阵L:", L0)
    print("训练后的矩阵L:", L)
    print('------------------------------------------------------------')
    
    print('------------------------总体效果------------------------------')
    print('训练集：')
    print("mean_absolute_error:", mean_absolute_error(train_yreal_array, train_yhat_array))
    print("mean_squared_error:", mean_squared_error(train_yreal_array, train_yhat_array))
    print("rmse:", sqrt(mean_squared_error(train_yreal_array, train_yhat_array)))
    print("r2 score:", r2_score(train_yreal_array, train_yhat_array))
    print('测试集：')
    print("mean_absolute_error:", mean_absolute_error(test_yreal_array, test_yhat_array))
    print("mean_squared_error:", mean_squared_error(test_yreal_array, test_yhat_array))
    print("rmse:", sqrt(mean_squared_error(test_yreal_array, test_yhat_array)))
    print("r2 score:", r2_score(test_yreal_array, test_yhat_array))
    print('------------------------------------------------------------')























