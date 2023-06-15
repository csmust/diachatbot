# coding: utf-8
import argparse
import json
# import pprint
from statistics import mean
# from time import sleep
import zipfile
import random
import torch
import os

from preprocess import preprocess
from train import train 
from sklearn.model_selection import RepeatedKFold
from collections import defaultdict
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def gen_kfold(k,data):
    rkf = RepeatedKFold(n_splits=k, n_repeats=1, random_state=2019)
    for train_index, test_index in rkf.split(data):
        train_data=[ data[i] for i in train_index]
        test_data=[ data[i] for i in test_index]
        with open(data_dir+"all_data_stric.json", 'r',encoding='utf-8') as f:
            augment_clean_data = json.load(f)
        #避免某些conversationId出现次数过多
        # random.shuffle(augment_clean_data)
        d = defaultdict(int)
        for augment_data in augment_clean_data:
            if augment_data['conversationId'] not in [ test_data_per['conversationId'] for test_data_per in test_data]:
                # if d[augment_data['conversationId']]<3:
                    train_data.append(augment_data)
                # d[augment_data['conversationId']]+=1

        yield  train_data , test_data


# list1=[ test_data_per['conversationId'] for test_data_per in test_data]
# print(list1)
# exit()
def dumpdata(train,test):
    dataname = ['val','train']   #val不是必须的
    splited_data = {
    "val_data" : test,
    "train_data" : train
    # "train_data" : data[60:]
    # "train_data" : data
    }

    for name in dataname:
        print('{}数据集数据量为：{}'.format(name,len(splited_data['{}_data'.format(name)])))
        
        split_json_file = data_dir + '{}.json'.format(name)
        with open(split_json_file, 'w', encoding='utf-8') as f:
            json.dump(splited_data['{}_data'.format(name)], f, ensure_ascii=False, sort_keys=True, indent=4)
        
        f = zipfile.ZipFile(split_json_file+'.zip','w',zipfile.ZIP_DEFLATED)#zipfile.ZIP_STORED不压缩
        f.write(split_json_file, arcname='{}.json'.format(name))
        f.close()
    print()


def cross_train(k_fold=10,filename="annotations_state_20220627_2.json",bertfile="",args=None):

    # 读入json文件
    best_val_F1_list=[]
    with open(data_dir+filename, 'r',encoding='utf-8') as f:
        data = json.load(f)
    print("数据集总长度{}".format(len(data)))
    gk=gen_kfold(k_fold,data)
    for k in range(k_fold):
        # 十次交叉验证

        print("*"*20+f"第{k+1}次交叉验证"+"*"*20)
        train_data, test_data=next(gk)

        dumpdata(train_data, test_data)

        if k==1:
            dataname = ['val','train']   
            splited_data = {
            "val_data" : test_data,
            "train_data" : train_data
            }

            for name in dataname:                
                split_json_file = data_dir + 'best_{}.json'.format(name)
                with open(split_json_file, 'w', encoding='utf-8') as f:
                    json.dump(splited_data['{}_data'.format(name)], f, ensure_ascii=False, sort_keys=True, indent=4)
                       
        preprocess('All',bertfile,CROSS_TRAIN=True)
        # if k !=1:
        #     continue
        # print(train_data[0])
        best_val_F1_list = train(True,best_val_F1_list,args)

    
    assert len(best_val_F1_list)==k_fold
    print(best_val_F1_list)
    print(mean(best_val_F1_list))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument('--config_path',
                    help='path to config file')
    args = parser.parse_args()
    config = json.load(open(args.config_path))
    bertpath=config["model"]["pretrained_weights"]

    set_seed(2019)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '../../../../data/diachat/')
    # cross_train(10,'annotations_state_20220627_2.json',bertpath,args)
    cross_train(10,'annotations_20220914_2.json',bertpath,args)
    # cross_train(10,'annotations_20221120_2.json',bertpath,args)


