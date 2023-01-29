#!/usr/bin/env python
# coding: utf-8

# # 「第2章 物体検出」の準備ファイル
# 
# - 本ファイルでは、第2章で使用するフォルダの作成とファイルのダウンロードを行います。

# In[1]:


import os
import urllib.request
import zipfile
import tarfile


# In[2]:


# フォルダ「data」が存在しない場合は作成する
#data_dir = "./data/"
#if not os.path.exists(data_dir):
   # os.mkdir(data_dir)
        

# In[3]:


# フォルダ「weights」が存在しない場合は作成する
weights_dir = "./weights/"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)
print("weight作成")

# In[6]:

# In[4]:


# 学習済みのSSD用のVGGのパラメータをフォルダ「weights」にダウンロード
# MIT License
# Copyright (c) 2017 Max deGroot, Ellis Brown
# https://github.com/amdegroot/ssd.pytorch
    
url = "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth"
target_path = os.path.join(weights_dir, "vgg16_reducedfc.pth") 
print("重みダウンロード")
if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)
print("ダウンロード終了")

# In[5]:


# 学習済みのSSD300モデルをフォルダ「weights」にダウンロード
# MIT License
# Copyright (c) 2017 Max deGroot, Ellis Brown
# https://github.com/amdegroot/ssd.pytorch

#url = "https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth"
#target_path = os.path.join(weights_dir, "ssd300_mAP_77.43_v2.pth") 

#if not os.path.exists(target_path):
   # urllib.request.urlretrieve(url, target_path)


# In[ ]:





# In[ ]:





# 以上
