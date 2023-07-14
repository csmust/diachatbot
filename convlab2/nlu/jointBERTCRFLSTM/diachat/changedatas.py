import os
import sys
import json

#获取当前文件目录名
curPath = os.path.dirname(__file__)
data_dir = os.path.join(curPath, 'data/All_data')
srcdate_key = ['train', 'val', 'test']
data={}
for key in srcdate_key:
    data[key]=json.load(open(os.path.join(data_dir, key + '_data.json'),encoding='utf-8') ) #,encoding='utf-8'
    print('load {}, size {}'.format(key, len(data[key])))
    f=open(os.path.join(data_dir, key + '_data.txt'), 'w', encoding='utf-8')
    for block in data[key]:
        zipdata=list(zip(block[0],block[1]))
        intent=""
        if len(block[2]) != 0:
            for i in range(len(block[2])):
                intent+=block[2][i]+"#"
        else:
            intent="none"
        intent=intent.rstrip("#")
        
        for i in range(len(zipdata)):
            if i >60:
                break
            word=zipdata[i][0]
            tag=zipdata[i][1]
            if tag!="O":
                tag=tag[:1]+"-"+tag[2:]
            f.write(word+" "+tag+"\n")
        f.write(intent+"\n"+"\n")
        # f.write(str(zipdata)+"\t"+intent+"\n")

    f.close()

        
                
            
        

