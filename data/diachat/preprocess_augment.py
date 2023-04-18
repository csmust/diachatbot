import os
import json
from collections import Counter
ACT = [
    "Inform",
    "AskForSure",
    "Deny",
    "Explanation",
    "Assure",
    "AskFor",
    "Advice",
    "GeneralAdvice",
    "Chitchat",
    "AdviceNot",
    "AskHow",
    "AskWhy",
    "Uncertain",
    "Accept",
    "GeneralExplanation"
]
DOMAIN = [
    "饮食",
    "问题",
    "行为",
    "",
    "治疗",
    "运动",
    "基本信息"
]
SLOT = [
    "饮食名",
    "血糖值",
    "行为名",
    "",
    "饮食量",
    "药品",
    "持续时长",
    "时间",
    "检查项",
    "检查值",
    "用药（治疗）频率",
    "用药量",
    "运动名",
    "疾病",
    "症状",
    "成分",
    "成份量",
    "频率",
    "效果",
    "治疗名",
    "部位",
    "体重",
    "身高",
    "症状部位",
    "状态",
    "年龄",
    "强度",
    "药品类型",
    "适应症",
    "既往史",
    "性别"
]
target_list = []
# 获取当前文件路径
# /yangjf/students/zhou/project/diachatbot/data/diachat/preprocess_augment.py
file = os.path.abspath(__file__)
# 获取当前文件所在目录
# /yangjf/students/zhou/project/diachatbot/data/diachat
path = os.path.dirname(file)
# print(path)
# 路径拼接
raw_file= os.path.join(path, 'annotations_20220914_2.json')
# /yangjf/students/zhou/project/diachatbot/data/diachat/Augument
augumentpath = os.path.join(path, 'Augment')
# print(augumentpath)
# 遍历文件夹
for root, dirs, files in os.walk(augumentpath):
    # print(root) #当前目录路径
    # print(dirs) #当前路径下所有子目录
    for dir in dirs:
        # print(dir)
        # print(os.path.join(root, dir)) #当前路径下所有子目录
        # 到子目录下遍历找到AugmentData.json
        for root1, dirs1, files1 in os.walk(os.path.join(root, dir)):
            # print(root1) #当前目录路径
            # print(dirs1) #当前路径下所有子目录
            # print(files1) #当前路径下所有非目录子文件
            for file in files1:
                if file == 'AugmentData.json':
                    print(os.path.join(root1, file))
                    target_list.append(os.path.join(root1, file))

            break
    # print(files) #当前路径下所有非目录子文件
    break
complete_id_list = []
all_data=[]
# conversation_dict=[]
i,j,k=0,0,0
m,n=0,0
outofdomain=[]
outofact=[]

rf=open(raw_file, 'r', encoding='utf-8')
raw_data = json.load(rf)
rf.close()

for target in target_list:
    with open(target, 'r', encoding='utf-8') as f:
        datalist = json.load(f)
        for conversation in datalist:
            flag=1
            id=conversation['conversationId']
            for seqid,turn in enumerate(conversation["utterances"]):

                #出现人工修改的话替换回原来的
                if turn=="人工修改":
                    i+=1
                    # flag=0
                    # break
                    #二分查找：找到对应的id
                    for conversation1 in raw_data:
                        if conversation1["conversationId"]==id:
                            turn=conversation1["utterances"][seqid]
                            conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                            break
                #严格清洗
                try:
                    jflag=0
                    kflag=0
                    for asv in turn["annotation"]:
                        if asv["act_label"] not in ACT:
                            outofact.append(asv["act_label"])
                            n+=1
                            for conversation1 in raw_data:
                                if conversation1["conversationId"]==id:
                                    # turn=conversation1["utterances"][seqid]
                                    conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                                    break
                            break
                            # flag=0
                            # break

                        for dsv in asv["slot_values"]:
                            
                            if dsv["domain"] =="None" or dsv["domain"] =="none" or  dsv["domain"] ==None:
                                # #替换成空字符串
                                # dsv["domain"]=""
                                # 改写价值不大，且增加标签类别，暂时不改
                                for conversation1 in raw_data:
                                    if conversation1["conversationId"]==id:
                                        # turn=conversation1["utterances"][seqid]
                                        conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                                        jflag=1
                                        break
                                if jflag==1:
                                    break
                            if dsv["slot"] =="None" or dsv["slot"] =="none" or dsv["slot"] ==None:
                                # 改写价值不大，且增加标签类别，暂时不改
                                for conversation1 in raw_data:
                                    if conversation1["conversationId"]==id:
                                        # turn=conversation1["utterances"][seqid]
                                        conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                                        jflag=1
                                        break
                                if kflag==1:
                                    break
                            if dsv["value"] =="None" or dsv["value"] =="none" or dsv["value"] ==None:
                                dsv["value"]=""
                            if (dsv["value"]=="" or dsv["value"]=="？") and (dsv["slot"]!="" and dsv["domain"]!=""):
                                dsv["value"]="?"
                            if dsv["domain"] not in DOMAIN:
                                outofdomain.append(dsv["domain"])
                                j+=1
                                for conversation1 in raw_data:
                                    if conversation1["conversationId"]==id:
                                        # turn=conversation1["utterances"][seqid]
                                        conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                                        jflag=1
                                        break
                                if jflag==1:
                                    break
                            if dsv["slot"] not in SLOT:
                                k+=1
                                for conversation1 in raw_data:
                                    if conversation1["conversationId"]==id:
                                        # turn=conversation1["utterances"][seqid]
                                        conversation["utterances"][seqid]=conversation1["utterances"][seqid]
                                        kflag=1
                                        break
                                if kflag==1:
                                    break
                        if jflag==1 or kflag==1:
                            break

                        # if dsv["value"]!="" and dsv["value"] not in turn["utterance"]:
                        #     flag=0
                        #     break
                except: 
                    for conversation1 in raw_data:
                        if conversation1["conversationId"]==id:
                            conversation["utterances"][seqid]=conversation1["utterances"][seqid]   
                            break
                    
            if (flag==1):
                all_data.append(conversation)
                complete_id_list.append(id)
                print(len(complete_id_list))
all_data.sort(key=lambda x: x["conversationId"])
complete_id_list.sort()
#统计complete_id_list中每个id出现的次数

c=Counter(complete_id_list)
c=dict(c)
c=sorted(c.items(),key=lambda x:x[1],reverse=True)
print(c)

f=open(os.path.join(path,"complete_id_list_stric.txt"),"w")
for id in complete_id_list:
    f.write(str(id))
    f.write("\n")
f.close()

with open(os.path.join(path,'all_data_stric.json'), 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)
print("需要人工修改计数=",i)
print("domain不在范围内计数=",j)
print("slot不在范围内计数=",k)
print("act不在范围内计数=",n)
 # 计算outofdomain中重复次数
c=Counter(outofdomain)
c=dict(c)
c=sorted(c.items(),key=lambda x:x[1],reverse=True)
print(c)
print()
# 计算outofact中重复次数
c=Counter(outofact)
c=dict(c)
c=sorted(c.items(),key=lambda x:x[1],reverse=True)
print(c)
            
                
                
