# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 18:51:31 2023
思路：神经网络 A L都可以训练

@author: lenovo
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
device = torch.device("cuda:0")	# 使用gpu训练

def sigmoid(x):
    
    y=1/(1+np.exp(-x))
    return y

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

# 生成数据读取器
def get_loader(data, seq_length, input_size, u_num, y_num, batch_size):
    """
    Params：
        seq_length：过去u、y时刻数
        input_size：u、y维度之和
        u_num：u的维度数
        y_num：y的维度数
        
    """
    data_spilo = data[:,:]
    #----------------------------选取过去u堆叠后的数据-----------------------------------
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
    #----------------------------选取过去y堆叠后的数据-----------------------------------
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
    #----------------------------选取一个时刻的u-------------------------------------
    u=[]
    for q in range(len(data_spilo)-seq_length-1):
        _u=data_spilo[q+seq_length,:u_num] # 选取u
        u.append(_u)
    u=np.array(u)
    #----------------------------选取下一时刻y（y1）-------------------------------------
    y1=[]
    for e in range(len(data_spilo)-seq_length-1): # range的范围需要减去时间步长和1
        _y=data_spilo[e+seq_length+1,-y_num:] # 选取y
        y1.append(_y)
    y1=np.array(y1)
    #----------------------------转为张量-----------------------------------
    x_data=Variable(torch.Tensor(np.array(x)))
    y_data=Variable(torch.Tensor(np.array(y)))
    u_data=Variable(torch.Tensor(np.array(u)))
    y1_data=Variable(torch.Tensor(np.array(y1)))
    #---------------------------放入读取器-----------------------------------
    dataset=Data.TensorDataset(x_data,y_data,u_data,y1_data)
    loader=torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False,drop_last=True)
      
    return loader

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
        self.layer1 = nn.Sequential(nn.Linear(self.input_num, self.hidden_num1), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(self.hidden_num1, self.hidden_num2), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(self.hidden_num2, self.output_num))
    def forward(self,input):
        input = self.layer1(input)
        input = self.layer2(input)
        output = self.layer3(input).to(device)
        return output
    
class My_loss(nn.Module):
    '''自定义损失函数'''
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        x=x.to(device)
        y=y.to(device)
        zhat = torch.mm(x, C.T)
        cha = (zhat - y).abs()
        return torch.mean(torch.pow(cha, 2))
    
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        # 定义可学习参数
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, loss1, loss2):
        # 限制alpha和beta的取值在0到1之间
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        # 构造总的归一化损失函数
        total_loss = ( alpha * loss1 + beta * loss2 ) / ( alpha + beta )
        return total_loss

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

# 得到训练数据data1 测试数据data2
data1,data2 = get_Data(path1,path2)
train_loader = get_loader(data1, seq_length, input_size, u_num, y_num, batch_size)
test_loader = get_loader(data2, seq_length, input_size, u_num, y_num, batch_size)

A = torch.rand(x_num, x_num).to(device)
A = torch.nn.Parameter(A.to(device).requires_grad_())
L = torch.rand(x_num, y_num).to(device)
L = torch.nn.Parameter(L.to(device).requires_grad_())

torch.save(A.data, './trained_model/train01/initial_A04.pt')  # 保存 原始的A
torch.save(L.data, './trained_model/train01/initial_L04.pt')  # 保存 原始的L
#A_expend_max,L_max=get_max(A, L, C, x_num, y_num, seq_length)
model = FCNNModel(input_num,hidden_num1,hidden_num2,output_num).to(device)
awl = CombinedLoss()
criterion1 = My_loss().to(device)
criterion2 = nn.MSELoss().to(device)
optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': [A,L],'lr': 1e-3}
            ], lr=8e-3)
# 打印训练之前的参数值
# print("初始参数 alpha: {:.4f}, beta: {:.4f}".format(awl.alpha.item(), awl.beta.item()))

train_losses = []
eval_losses = []
for e in range(200):
    train_loss = 0
    model.train()   # 将模型改为训练模式
    # 每次迭代都是处理一个小批量的数据，batch_size是16 for i,(batch_x, batch_u, batch_y) in enumerate (train_loader):
    for batch_x, batch_y, batch_u, batch_y1 in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device) 
        batch_u = batch_u.to(device)
        batch_y1 = batch_y1.to(device)
        
        A_expend_max,L_max=get_max(A, L, C, x_num, y_num, seq_length)
        # A_expend1 = A - torch.mm(L, C)
        
        # 将batch_x先切片为10份，再分别放入神经网络
        chunks = torch.chunk(batch_x, seq_length, 1) 
        new_chunks = [model(chunk) for chunk in chunks]
        result1 = torch.cat(new_chunks, dim=1)
        state11 = torch.mm(result1, A_expend_max)
        state12 = torch.mm(batch_y, L_max)
        state1 = torch.add(state11, state12) # xk 前半段的状态
        
        state21 = torch.mm(state1, A.T)
        state22 = model(batch_u)
        state2 = torch.add(state21, state22) # xk+1 后半段的状态
        
        loss1 = criterion1(state2, batch_y1)
        outputs1_cut,outputs2_cut = state1[1:],state2[:-1]
        loss2 = criterion2(outputs1_cut, outputs2_cut)
        loss = loss1 + loss2 + torch.norm(torch.pow(A-torch.mm(L, C), seq_length+1), p=2)**2
        
        optimizer.zero_grad()
        # 反向传播，记得要把上一次的梯度清0，反向传播，并且step更新相应的参数。
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))
        # 在测试集上检验效果
    eval_loss = 0
    A_expend_max,L_max=get_max(A, L, C, x_num, y_num, seq_length)
    model.eval()  # 将模型改为预测模式
    for batch_x, batch_y, batch_u, batch_y1 in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device) 
        batch_u = batch_u.to(device)
        batch_y1 = batch_y1.to(device)
        
        # 将batch_x先切片为10份，再分别放入神经网络
        chunks = torch.chunk(batch_x, seq_length, 1) 
        new_chunks = [model(chunk) for chunk in chunks]
        result1 = torch.cat(new_chunks, dim=1)
        state11 = torch.mm(result1, A_expend_max)
        state12 = torch.mm(batch_y, L_max)
        state1 = torch.add(state11, state12) # xk 前半段的状态
        
        state21 = torch.mm(state1, A.T)
        state22 = model(batch_u)
        state2 = torch.add(state21, state22) # xk+1 后半段的状态
        
        loss1 = criterion1(state2, batch_y1)
        outputs1_cut,outputs2_cut = state1[1:],state2[:-1]
        loss2 = criterion2(outputs1_cut, outputs2_cut)
        loss = loss1 + loss2 + torch.norm(torch.pow(A - torch.mm(L, C), seq_length+1), p=2)**2
        
        optimizer.zero_grad()
        # 记录误差
        eval_loss += loss.item()
    eval_losses.append(eval_loss / len(test_loader))
    print('epoch: {}, Train Loss: {:.6f},Eval Loss: {:.6f}'
          .format(e, train_loss / len(train_loader),eval_loss / len(test_loader)))
    
# 存储参数
torch.save(model.state_dict(), './trained_model/train01/model04.pt')
torch.save(A.data, './trained_model/train01/trained_A04.pt')  # 保存 A
torch.save(L.data, './trained_model/train01/trained_L04.pt')  # 保存 L
# 存储整个模型
# torch.save(model, './trained_model/model01.pt')
# 打印训练之后的参数值
# print("训练后参数 alpha: {:.4f}, beta: {:.4f}".format(awl.alpha.item(), awl.beta.item()))













