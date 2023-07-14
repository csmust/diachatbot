import sys
import os
import json

#获取当前文件目录名
curPath = os.path.dirname(__file__)



def change(type):
    data_dir = curPath
    data_dict=json.load(open(os.path.join(data_dir, type + '.json'),encoding='utf-8') ) #,encoding='utf-8'
    print('load {}'.format(type))
    f=open(os.path.join(data_dir, type + '.txt'), 'w', encoding='utf-8')
    for id,block_dict in data_dict.items():
        block_list=block_dict['dialogue']
        for sentence_dict in block_list:
            sentence=sentence_dict["speaker"]+"："+sentence_dict['sentence']
            BIO_label=("O O O "+sentence_dict['BIO_label']).split(" ")
            # print(BIO_label)
            zipdata=list(zip(sentence,BIO_label))
            assert len(sentence)==len(zipdata)
            for i in range(len(zipdata)):
                # if i >60:
                #     break
                word=zipdata[i][0]
                tag=zipdata[i][1]
                f.write(word+" "+tag+"\n")
            f.write(sentence_dict["dialogue_act"]+"\n"+"\n")
if __name__ == '__main__':
    data_types=['train', 'val', 'test']
    for type in data_types:
        change(type)