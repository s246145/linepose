import os.path as osp
import random
import time

import importlib
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from scipy.spatial.transform import Rotation
from pytorch_metric_learning import miners, losses
#from utils.ssd_model import make_datapath_list, VOCDataset, DataTransform, Anno_xml2list, od_collate_fn,MultiBoxLoss
import utils.ssd_model_detect
import importlib
from apex import amp
import apex
importlib.reload(utils.ssd_model_detect) #importと同じモジュールを指定する。
import os
from utils.ssd_model_detect import make_datapath_list, VOCDataset, DataTransform, Anno_xml2list, od_collate_fn,MultiBoxLoss
import math
# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# ファイルパスのリストを取得
rootpath = "/home/s246145/lmo"
train_img_list, train_anno_list,train_pose_list,val_img_list, val_anno_list,val_pose_list = make_datapath_list(
    rootpath)
#print(val_img_list)
#train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
#    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes),anno_pose_list=train_pose_list)

#val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
#   input_size, color_mean), transform_anno=Anno_xml2list_test(voc_classes),anno_pose_list=val_pose_list)
print("ファイルパスリスト取得")
# # DatasetとDataLoaderを作成する
batch_size = 1
print(batch_size)

np.set_printoptions(threshold=10)
# Datasetを作成
voc_classes = ['1','2','3','4','5','6','7','8','9','10',
               '11','12','13','14','15']
color_mean = (120, 120, 120)  # (BGR)の色の平均値
input_size = 300  # 画像のinputサイズを300×300にする

train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
    input_size, color_mean), transform_anno=Anno_xml2list(voc_classes),anno_pose_list=train_pose_list)

val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
   input_size, color_mean), transform_anno=Anno_xml2list(voc_classes),anno_pose_list=val_pose_list)

train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, num_workers=20,collate_fn=od_collate_fn,pin_memory=True)

val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=20,collate_fn=od_collate_fn,pin_memory=True)
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
print("dataset作成")

print("dataLoader作成")
#print(len(val_dataset))
# # ネットワークモデルの作成する

# In[62]:


from utils.ssd_model_detect import SSD
#from utils.ssd_model_1 import SSD
# SSD300の設定
ssd_cfg = {
    'num_classes': 16,  # 背景クラスを含めた合計クラス数
    'input_size': 300,  # 画像の入力サイズ
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

# SSDネットワークモデル
net = SSD(phase="train", cfg=ssd_cfg)
net.eval()

# SSDの初期の重みを設定
# ssdの全部分の重みをロード
#net_weights = torch.load('./weights/ssd300_linemod_1-3e3.pth')
#net.load_state_dict(net_weights)

checkpoint = torch.load('./last4e-5_checkpoint+50.pt')
#model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
net.load_state_dict(checkpoint['model'])
net.to('cuda:0')

# GPUが使えるかを確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

#import utils.ssd_model
#import importlib
#importlib.reload(utils.ssd_model) #importと同じモジュールを指定する。
#from utils.ssd_model import make_datapath_list, VOCDataset, DataTransform, Anno_xml2list, od_collate_fn,MultiBoxLoss
torch.set_printoptions(edgeitems=1000)
#criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)
#line,conf = criterion(outputs, targets)
def line_model(net, dataloaders_dict, criterion):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    i = 0
    A = 0
    B = 0
    c = []
    for phase in ['train', 'val']:
        if phase == 'train':
            net.eval()  # モデルを訓練モードに
                #scaler = torch.cuda.amp.GradScaler(growth_interval=100)

            print('eval')
        else:
            continue

            # データローダーからminibatchずつ取り出すループ
        for images, targets in dataloaders_dict[phase]:

                # GPUが使えるならGPUにデータを送る
            images = images.to(device,non_blocking=True)
            targets = [ann.to(device,non_blocking=True)
                        for ann in targets]  # リストの各要素のテンソルをGPUへ

                # optimizerを初期化

                # 順伝搬（forward）計算
                #with torch.set_grad_enabled(phase == 'train'):
    
            outputs = net(images)
            line,conf= criterion(outputs, targets)
            line = line.to('cpu').detach().numpy().copy()
            conf = conf.to('cpu').detach().numpy().copy()
            #print(line)
            #print(conf)
            conf = conf[:,np.newaxis]
            if i == 0:
                A = np.concatenate([line, conf], 1)
            else:
                C = np.concatenate([line, conf], 1)
                B = np.concatenate([A, C])
                A = B

            if i == 50000:
                break
            i = i + 1
            print(i)

    return A
#criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)
#A = line_model(net=net, dataloaders_dict=dataloaders_dict, criterion=criterion)
#np.savetxt('linedata_095.txt', A)

def line_model_test(net, dataloaders_dict, criterion):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    i = 0
    A = 0
    B = 0
    c = []
    for phase in ['train', 'val']:
        if phase == 'train':
            continue
            
        else:
            net.eval()  # モデルを訓練モードに
                #scaler = torch.cuda.amp.GradScaler(growth_interval=100)

            print('eval')

            # データローダーからminibatchずつ取り出すループ
        for images, targets in dataloaders_dict[phase]:

                # GPUが使えるならGPUにデータを送る
            images = images.to(device,non_blocking=True)
            targets = [ann.to(device,non_blocking=True)
                        for ann in targets]  # リストの各要素のテンソルをGPUへ

                # optimizerを初期化

                # 順伝搬（forward）計算
                #with torch.set_grad_enabled(phase == 'train'):

            outputs = net(images)
            line,conf= criterion(outputs, targets)
            line = line.to('cpu').detach().numpy().copy()
            conf = conf.to('cpu').detach().numpy().copy()
            #print(line)
            #print(conf)
            conf = conf[:,np.newaxis]
            if i == 0:
                A = np.concatenate([line, conf], 1)
            else:
                C = np.concatenate([line, conf], 1)
                B = np.concatenate([A, C])
                A = B

            if i == 10000:
                break
            i = i + 1
            print(i)

    return A
linedata = np.loadtxt('./linedata_095.txt')
linedata = np.array(linedata)

criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)
linetest = line_model_test(net=net, dataloaders_dict=dataloaders_dict, criterion=criterion)
#np.savetxt('linedata_095.txt', A)
print(linedata.shape)
#print(B.shape)
line_label = []
L = 0
B = 0
for i in range(linetest.shape[0]):
    D = abs((linedata[0:,0:32] - linetest[i,0:32]).sum(axis = 1))
#print(D.shape)
    min_val = np.min(D)#最小値の
    LINE = linedata[np.where(D == min_val)]
#print(LINE)
#print(LINE.shape)
    line_label.append(LINE[0][32])
    print(i)
    if linetest[i,32] == line_label[L]:
        B = B + 1
        L = L + 1
    else:
        L = L + 1
print("種類推定精度"+str(B/L))
#linedata = np.loadtxt('./linedata_new.txt')
#print(len(A)) 
#def ssd_lined(img_index, img_list, dataset, net=None, dataconfidence_level=0.7):

def ssd_predict(img_index, img_list, dataset, net=None, dataconfidence_level=0.7):

    """
    SSDで予測させる関数。

    Parameters
    ----------
    img_index:  int
        データセット内の予測対象画像のインデックス。
    img_list: list
        画像のファイルパスのリスト
    dataset: PyTorchのDataset
        画像のDataset
    net: PyTorchのNetwork
        学習させたSSDネットワーク
    dataconfidence_level: float
        予測で発見とする確信度の閾値

    Returns
    -------
    rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
    """

    # rgbの画像データを取得
    image_file_path = img_list[img_index]
    img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
    height, width, channels = img.shape  # 画像のサイズを取得
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 正解のBBoxを取得
    im, gt = dataset.__getitem__(img_index)
    true_bbox = gt[:, 0:4] * [width, height, width, height]
    true_label_index = gt[:, 4].astype(int)
    true_pose = gt[:,5:9]
    true_ignore = gt[:,9].astype(int)
    j = 0
    true_bbox_i =[]
    true_label_index_i = []
    true_pose_i = []
    for i in range(len(true_ignore)):
        if true_ignore[i] == 0:
            true_bbox_i.append(true_bbox[i])
            true_label_index_i.append(true_label_index[i])
            true_pose_i.append(true_pose[i])
            j = j + 1
    
    #print(true_bbox_i)
    # SSDで予測
    net.eval()  # ネットワークを推論モードへ
    x = im.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 300, 300])
    detections = net(x)
    # detectionsの形は、torch.Size([1, 21, 200, 41])  ※200はtop_kの値

    # confidence_levelが基準以上を取り出す
    predict_bbox = []
    pre_dict_label_index = []
    scores = []
    poses = []
    lines = []
    detections = detections.cpu().detach().numpy()

    # 条件以上の値を抽出
    find_index = np.where(detections[:, 0:, :, 0] >= dataconfidence_level)
    detections = detections[find_index]
    for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す
        if (find_index[1][i]) > 0:  # 背景クラスでないもの
            sc = detections[i][0]  # 確信度
            bbox = detections[i][1:5] * [width, height, width, height]
            line = detections[i][5:37]
            pose = detections[i][37:41]
            lable_ind = find_index[1][i]-1  # find_indexはミニバッチ数、クラス、topのtuple
            # （注釈）
            # 背景クラスが0なので1を引く

            # 返り値のリストに追加
            predict_bbox.append(bbox)
            pre_dict_label_index.append(lable_ind)
            scores.append(sc)
            pose[0:3] = pose[0:3] * np.sin(pose[3])
            pose[3] = np.cos(pose[3]/2)
            pose_norm = np.linalg.norm(pose,ord=2)
            pose = pose / pose_norm
            poses.append(pose)
            lines.append(line)
            
            
    return rgb_img, true_bbox_i, true_label_index_i, true_pose_i,predict_bbox, pre_dict_label_index, scores,poses,lines

# = 6
#rgb_img, true_bbox, true_label_index, true_pose,predict_bbox,pre_dict_label_index, scores,poses,lines = ssd_predict(img_index = i,
#            img_list = val_img_list,dataset = val_dataset, net=net, dataconfidence_level=0.5)
#for i in range(len(val_img_list)):
    #rgb_img, true_bbox, true_label_index, true_pose,predict_bbox,pre_dict_label_index, scores,poses,lines = ssd_predict(img_index = i,
            #img_list = val_img_list,dataset = val_dataset, net=net, dataconfidence_level=0.5)
    #print(true_pose)
    #print(poses)
    #print(true_label_index)
    #print(pre_dict_label_index)

from scipy.spatial.transform import Rotation as R

def ssd_pose_lost(true_bbox,predict_bbox,true_pose,poses,line,true_label_index):
    t_bbox = []
    t_pose = []
    t_line = []
    t_label = []
    for i,pbox in enumerate(predict_bbox):
        distance = []
        for j , tbox in enumerate(true_bbox):
            distance.append(abs(sum(predict_bbox[i]-true_bbox[j])))#tとpの距離を比較
        if len(distance) != 0:
            #print(distance.index(min(distance)))
            #print(line[0])
            #print(len(true_bbox))
            #print(len(line))
            #print(len(true_label_index))
            t_bbox.append(true_bbox[distance.index(min(distance))])
            t_pose.append(true_pose[distance.index(min(distance))])
            #t_line.append(line[distance.index(min(distance))])
            t_label.append(true_label_index[distance.index(min(distance))])
            
            #print(len(true_label_index))
    return t_bbox,t_pose,t_label
     
def create_matrix(pose):
    pose = R.from_quat(pose)
    pose = pose.as_matrix()
    return(pose)
#print(ssd_pose_lost(true_bbox,predict_bbox,true_pose,poses))

def create_euler(pose):
    pose = R.from_quat(pose)
    pose = pose.as_euler('zyx',degrees = True)
    return(pose)

def create_matrix_p(poset,pose,rotation_z):
    poset = R.from_quat(poset)
    poset = poset.as_matrix()
    rot_z = np.array([
        [math.cos(math.radians(rotation_z)),-math.sin(math.radians(rotation_z)),0],
        [math.sin(math.radians(rotation_z)),math.cos(math.radians(rotation_z)),0],
        [0,0,1]
        ])
    poset = np.dot(poset,rot_z)
    poset = R.from_matrix(poset)
    poset = poset.as_quat()
    #print(math.degrees(2 * np.arccos(np.dot(poset,pose))))
    return(poset)
def iou(a, b):
    # a, bは矩形を表すリストで、a=[xmin, ymin, xmax, ymax]
    ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
    bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w*h

    iou = intersect / (a_area + b_area - intersect)
    return iou

A = 0
P = 0
B = 0
L = 0
U = 0
O = 0
E = 0
#lines,conf = ssd_lined(img_index = 0,
#            img_list = train_img_list,dataset = train_dataset, net=net, dataconfidence_level=0.4)
#print(lines,conf)
net = SSD(phase="inference", cfg=ssd_cfg)
net.eval()

# SSDの初期の重みを設定
# ssdの全部分の重みをロード
#net_weights = torch.load('./weights/ssd300_linemod_1-3e3.pth')
#net.load_state_dict(net_weights)

checkpoint = torch.load('./last4e-5_checkpoint+50.pt')
#model, optimizer = amp.initialize(net, optimizer, opt_level=opt_level)
net.load_state_dict(checkpoint['model'])

# GPUが使えるかを確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)
linedata = np.loadtxt('./linedata_095.txt')
linedata = np.array(linedata)
print(linedata.shape)
#print(linedata)
line_label = []

for i in range(len(val_img_list)):
    rgb_img, true_bbox, true_label_index, true_pose,predict_bbox,pre_dict_label_index, scores,poses,lines = ssd_predict(img_index = i,
            img_list = val_img_list,dataset = val_dataset, net=net, dataconfidence_level=0.4)
    #print(scores)
    #print(lines)
    #print(lines)
    t_bbox,t_pose,t_label = ssd_pose_lost(true_bbox,predict_bbox,true_pose,poses,lines,true_label_index)
    t_pose = np.array(t_pose)
    poses = np.array(poses)
    lines = np.array(lines)
    print(i)
    #print(t_bbox)
    #print(t_pose)
    #print(t_line)
    #print(t_label)
    #if poses.size != 0:
        #t_pose = R.from_quat(t_pose)
        #t_pose = t_pose.as_euler('zyx',degrees = True)
        #poses = R.from_quat(poses)
        #t_pose = poses.as_euler('zyx',degrees = True)
    #    for i,(t,p) in enumerate(zip(t_pose,poses)):
            #create_matrix_p(t,p,180)
            #print(create_euler(t))
            #print(create_euler(p))
            #print(math.degrees(2 * np.arccos(np.dot(t,p))))
    #        if math.degrees(2 * np.arccos(np.dot(t,p))) > 0:
    #            A = A + math.degrees(2 * np.arccos(np.dot(t,p)))
    #            P = P + 1
    #line_label = []
    #if len(lines) != 0:
    #    for i in zip(lines):
    #        distance = []
    #        for j in zip(linedata):
    #            J = j[0][:32]
    #            distance.append(sum((abs(sum(np.asarray(i)-np.asarray(J)))))) #print(distance)
            #abs(sum(predict_bbox[i]-true_bbox[j]))
            
    #        LINE = linedata[distance.index(min(distance))]
    #        line_label.append(LINE[32])
    #        #print(line_label)
    #    for i , k in enumerate(t_label):
    #        if k == line_label[i]:
    #            B = B + 1
    #            L = L + 1
    #        else:
    #            L = L + 1
    #print(predict_bbox)
    if len(predict_bbox) != 0:
        for i in range(len(predict_bbox)):
            I = iou(predict_bbox[i],t_bbox[i])
            #print(I)
            #D = abs((linedata[0:,0:32] - lines[i]).sum(axis = 1))
            #LINE = linedata[D.index(min(D))]
            #line_label.append(LINE[32])
            if I > 0.75:
                                
                #distance = []
                #print(linedata.shape())
                D = abs((linedata[0:,0:32] - lines[i]).sum(axis = 1))
                print(D.shape)
                min_val = np.min(D)
                LINE = linedata[np.where(D == min_val)]
                print(LINE)
                print(LINE.shape)
                line_label.append(LINE[0][32])
                #for j in zip(linedata):
                  
                #    J = j[0][:32]
                    #print(J)
                #    D = (abs(sum(np.asarray(lines[i])-np.asarray(J))))
                #    distance.append(D)
                    #print(np.asarray(lines[i]))
                    #print(np.asarray(J))
                    
                #LINE = linedata[distance.index(min(distance))]
                #line_label.append(LINE[32])
                #print(line_label)
                #t_label[i]
                #line_label[L]
                if t_label[i] == line_label[L]:
                    B = B + 1
                    L = L + 1
                else:
                    L = L + 1
        
                A = A + math.degrees(2 * np.arccos(np.dot(t_pose[i],poses[i])))
                P = P + 1

                O = O + 1
            else:
                O = O
    #print(O)
    U = U + len(true_bbox)
    #print(U)
    #print("観測:"+str(pre_dict_label_index))
    #print("観測:"+str(predict_bbox))
    #print("真実:"+str(true_label_index))
    #print("真実:"+str(true_bbox))

#    print("観測:"+str(pre_dict_label_index))
print("回転誤差"+str(A/P))
print("種類推定精度"+str(B/L))
print("IOU率"+str(O/U))
