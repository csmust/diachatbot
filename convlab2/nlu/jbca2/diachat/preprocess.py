# 导入模块
import json
import os
import zipfile
import sys
from collections import Counter
from transformers import BertTokenizer







# 解压函数
def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


# 预处理函数
def preprocess(mode,tokenizerpath,CROSS_TRAIN=False):
    assert mode == 'All' or mode == 'User' or mode == 'Doctor' 
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '../../../../data/diachat/')
    processed_data_dir = os.path.join(cur_dir, 'data/{}_data'.format(mode))

    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    data_key = ['train', 'val', 'test']
    if CROSS_TRAIN :
        data_key=['train', 'val']
    data = {}
    for key in data_key:
        data[key] = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        print('load {}, size {}'.format(key, len(data[key])))

    processed_data = {}
    all_intent = []
    all_tag = []
    all_act=[]
    all_domain=[]

    context_size = 3

    if tokenizerpath:
        try:
            tokenizer = BertTokenizer.from_pretrained(tokenizerpath)  # tokenizerpath就是bert模型所在路径，这里就是自己训练的bert模型所在路径
            print("tokenizer的路径为:{}".format(tokenizerpath))
        except:
            print("请传入正确的tokenizerpath,比如python .\preprocess.py E:/Local-Data/MedDialogueGenNew/output/mlm/01/model不传入则默认为hfl/chinese-bert-wwm-ext")
    else:
#         try:
#             tokenizer = BertTokenizer.from_pretrained("E:/Local-Data/models_datasets/chinese-bert-wwm-ext") # remote108
#             print("remote108 tokenizer E:/Local-Data/models_datasets/chinese-bert-wwm-ext ")
#         except:
        '''
        hfl/chinese-bert-wwm-ext 是 https://huggingface.co/hfl 页面 名为hfl/chinese-bert-wwm-ext的bert模型
        BertTokenizer可自行下载并加载
                    也可直接在页面https://huggingface.co/hfl/chinese-bert-wwm-ext下载
        '''
        tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    for key in data_key:
        processed_data[key] = []
        for sess in data[key]:
            context = []
            for i, turn in enumerate(sess['utterances']):
                if mode == 'User' and turn['agentRole'] == 'Doctor':
                    context.append(turn['utterance'])
                    continue
                elif mode == 'Doctor' and turn['agentRole'] == 'User':
                    context.append(turn['utterance'])
                    continue
                utterance = turn['utterance']
                # Notice: ## prefix, space remove
                tokens = tokenizer.tokenize(utterance)

                golden = []

                span_info = []
                intents = []
                domain=[]
                act=[]

                for i in turn['annotation']:
                    value2index={}
                    for j in i['slot_values']:
                        if j['value'] is not None and j['value'] != '' and j['value'] != '？' and j['value'] != '?':
                            if j['value'] in utterance:
                                # if j['value'] not in value2index:
                                #     idx = utterance.index(j['value'])   # 有点小bug  value【吃药】 在utterance中重复出现时候比如 "我现在没有吃药，不需要吃药吗？" 我们需要标注后一个吃药时
                                # else:
                                #     idx = utterance.index(j['value'],value2index[j['value']]+1)
                                # value2index[j['value']]=idx
                                idx = utterance.index(j['value'])  
                                idx = len(tokenizer.tokenize(utterance[:idx]))
                                span_info.append((
                                    '+'.join([i['act_label'], j['domain'], j['slot']]), idx,
                                    idx + len(tokenizer.tokenize(j['value'])),
                                    j['value']))
                                token_v = ''.join(tokens[idx:idx + len(tokenizer.tokenize(j['value']))])
                                # if token_v != j['value']:    #
                                #     print("--- token_v != j['value'] ---")    # '5—7' 应该为"5-7"  短杠问题 token 被记为 5[UNK]7
                                #     print(j['slot'], token_v, j['value'],"\n")
                                # if "##" in token_v:
                                #     print("--- ## in token_v ---")
                                #     print(j['slot'], token_v, j['value'],"\n")
                                token_v = token_v.replace('##', '')
                                golden.append([i['act_label'], j['domain'], j['slot'], token_v])
                                domain.append(j['domain'])
                                act.append(i['act_label'])
                            else:
                                # print("--- value不在utterance中 ---")
                                # print(j['value'], utterance,"\n")
                                golden.append([i['act_label'], j['domain'], j['slot'], j['value']])
                                domain.append(j['domain'])
                                act.append(i['act_label'])
                        else:
                            intents.append('+'.join([i['act_label'], j['domain'], j['slot'], j['value']]))
                            golden.append([i['act_label'], j['domain'], j['slot'], j['value']])
                            domain.append(j['domain'])
                            act.append(i['act_label'])

                tags = []
                for j, _ in enumerate(tokens):
                    for span in span_info:
                        if j == span[1]:
                            tag = "B+" + span[0]
                            tags.append(tag)
                            break
                        if span[1] < j < span[2]:
                            tag = "I+" + span[0]
                            tags.append(tag)
                            break
                    else:
                        tags.append("O")

                processed_data[key].append([tokens, tags, intents, golden, context[-context_size:]])
                # print([tokens, tags, intents, golden, context[-context_size:]])
                # input()

                all_intent += intents
                all_tag += tags
                all_domain+=domain
                all_act+=act

                context.append(turn['utterance'])

        all_intent = [x[0] for x in dict(Counter(all_intent)).items()]
        all_tag = [x[0] for x in dict(Counter(all_tag)).items()]
        all_domain=[x[0] for x in dict(Counter(all_domain)).items()]
        all_act=[x[0] for x in dict(Counter(all_act)).items()]


        print('loaded {}, size {}'.format(key, len(processed_data[key])))
        json.dump(processed_data[key],
                  open(os.path.join(processed_data_dir, '{}_data.json'.format(key)), 'w', encoding='utf-8'),
                  indent=2, ensure_ascii=False)

    print('sentence label num:', len(all_intent))
    print('tag num:', len(all_tag))
    print(all_intent)
    json.dump(all_intent, open(os.path.join(processed_data_dir, 'intent_vocab.json'), 'w', encoding='utf-8'), indent=2,
              ensure_ascii=False)
    json.dump(all_tag, open(os.path.join(processed_data_dir, 'tag_vocab.json'), 'w', encoding='utf-8'), indent=2,
              ensure_ascii=False)
    json.dump(all_domain, open(os.path.join(processed_data_dir, 'domain_vocab.json'), 'w', encoding='utf-8'), indent=2,
              ensure_ascii=False)
    json.dump(all_act, open(os.path.join(processed_data_dir, 'act_vocab.json'), 'w', encoding='utf-8'), indent=2,
              ensure_ascii=False)

if __name__ == '__main__':
    '''
  该参数与config文件的 pretrained_weights参数值保持一致。比如
    “python preprocess.py E:/Local-Data/models_datasets/chinese-bert-wwm-ext”, 
    不加参数默认为原hfl/chinese-bert-wwm-ext
   模型训练和加载的时候选用对应的config文件，以便加载对那个的bert模型
    '''
    path=""
    if len(sys.argv)>1:
        path=sys.argv[1]
    # preprocess('User',path)
    preprocess('All',path)
    # preprocess('Doctor',path)
