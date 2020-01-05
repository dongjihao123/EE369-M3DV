# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 11:09:25 2020

@author: DELL
"""

import numpy as np
import pandas as pd
import os,csv
from tqdm import tqdm

def crop(array, zyx, dhw):
    z, y, x = zyx
    d, h, w = dhw
    cropped = array[z - d // 2:z + d // 2,
              y - h // 2:y + h // 2,
              x - w // 2:x + w // 2]
    return cropped

for root, dirs,files in os.walk('test'):   
    filename = files

def sort_key(s):
    return int(s[9:-4])

test_number = sorted(filename,key = sort_key)         #获得排序的测试集

#读取测试数据
voxel_test = []     #用于存储测试数据的voxel
seg_test = []       #用于存储测试数据的seg

for i in tqdm(range(584), desc='reading test_data'):    #写入测试数据的进度
    try:
        tmp = np.load('test/candidate{}.npz'.format(i)) #依次读取测试数据中的candidate{i}文件
    except FileNotFoundError:                           #无该文件时直接进入下一次循环
        continue
    try:
        voxel_test = np.append(voxel_test, np.expand_dims(tmp['voxel'], axis=0), axis=0)    #向voxel_test中添加读取的voxel向量
        seg_test = np.append(seg_test, np.expand_dims(tmp['seg'], axis=0), axis=0)          #向seg_test中添加读取的seg向量
    except ValueError:
        voxel_test = np.expand_dims(tmp['voxel'], axis=0)   #写入初次读取的voxel
        seg_test = np.expand_dims(tmp['seg'], axis=0)       #写入初次读取的seg

seg_test = seg_test.astype(np.int)      #将seg布尔array转换为1/0整数
X_test= voxel_test*seg_test             #抠取结节

X_test=X_test.astype(np.float32)
X_test/=128.-1.

test_batch_size = X_test.shape[0]  #测试数据集的数量

X_test_new=crop(X_test[0],(50,50,50),(32,32,32))

X_test_new=np.expand_dims(X_test_new,axis=0)

for i in tqdm(range(test_batch_size-1),desc='croping'):
    X_test_new=np.append(X_test_new,np.expand_dims(crop(X_test[i+1],(50,50,50),(32,32,32)),axis=0),axis=0)   
del X_test
X_test_new = X_test_new.reshape(X_test_new.shape[0], 32, 32, 32, 1)     #将测试数据集整合成5d张量


from keras.models import load_model
#载入模型 
model = load_model('model.h5')

#预测并存储结果
 
y_pred=model.predict(X_test_new)

test_label = []
test_label.append(['id','Predicted'])
for i in range(test_batch_size):
    test_label.append([test_number[i][:-4],y_pred[i][1]])
    
with open('Submission.csv', 'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerows(test_label)