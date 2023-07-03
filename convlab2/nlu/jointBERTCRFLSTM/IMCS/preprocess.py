# 导入模块
import json
import os
import zipfile
import sys
from collections import Counter
from transformers import BertTokenizer
from transformers import AutoTokenizer

tags = [
    'Request-Etiology', 'Request-Precautions', 'Request-Medical_Advice', 'Inform-Etiology', 'Diagnose',
    'Request-Basic_Information', 'Request-Drug_Recommendation', 'Inform-Medical_Advice',
    'Request-Existing_Examination_and_Treatment', 'Inform-Basic_Information', 'Inform-Precautions',
    'Inform-Existing_Examination_and_Treatment', 'Inform-Drug_Recommendation', 'Request-Symptom',
    'Inform-Symptom', 'Other'
]
tag2id = {tag: idx for idx, tag in enumerate(tags)}


def make_data(samples, path):
    out = ''
    for pid, sample in samples.items():
        for sent in sample['dialogue']:
            x = sent['speaker'] + '：' + sent['sentence']
            assert sent['dialogue_act'] in tag2id
            y = tag2id.get(sent['dialogue_act'])
            out += x + '\t' + str(y) + '\n'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(out)
    return out

def save_data(samples, input_fn, output_fn2):
    seq_in = []
    seq_bio = []
    for pid, sample in samples.items():
        for item in sample['dialogue']:
            sent = list(item['speaker'] + '：' + item['sentence'])
            bio = ['O'] * 3 + item['BIO_label'].split(' ')
            assert len(sent) == len(bio)
            seq_in.append(sent)
            seq_bio.append(bio)
    assert len(seq_in) == len(seq_bio)
    print('句子数量为：', len(seq_in))
    with open(input_fn, 'w', encoding='utf-8') as f1:
        for i in seq_in:
            tmp = ' '.join(i)
            f1.write(tmp + '\n')
    with open(output_fn2, 'w', encoding='utf-8') as f2:
        for i in seq_bio:
            tmp = ' '.join(i)
            f2.write(tmp + '\n')


def tag2das(word_seq, tag_seq):
    assert len(word_seq)==len(tag_seq)
    das = []
    i = 0
    while i < len(tag_seq):
        tag = tag_seq[i]
        if tag.startswith('B'):
            slot = tag[2:].split('-')[-1]
            intent=""
            domain=""
            value = word_seq[i]
            j = i + 1
            while j < len(tag_seq):
                if tag_seq[j].startswith('I') and tag_seq[j][2:] == tag[2:]:
                    # tag_seq[j][2:].split('+')[-1]==slot or tag_seq[j][2:] == tag[2:]
                    if word_seq[j].startswith('##'):
                        value += word_seq[j][2:]
                    else:
                        value += word_seq[j]
                    i += 1
                    j += 1
                else:
                    break
            if value.startswith('##'):
                value=value.replace('##','')
            das.append([intent, domain, slot, value])
        i += 1
    return das

# 预处理函数
def preprocess(mode,tokenizerpath,CROSS_TRAIN=False):
    assert mode == 'All' or mode == 'User' or mode == 'Doctor' 
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '../../../../data/IMCS/')
    processed_data_dir = os.path.join(cur_dir, 'data/{}_data'.format(mode))

    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    data_key = ['train', 'val', 'test']
    if CROSS_TRAIN :
        data_key=['train', 'val']
    data = {}
    for key in data_key:
        data[key]=json.load(open(os.path.join(data_dir, '{}.json'.format(key)), 'r', encoding='utf-8'))
        print('load {}, size {}'.format(key, len(data[key])))

    processed_data = {}
    all_intent = []
    all_tag = []
    all_act=[]
    all_domain=[]
    all_golden=[]
    all_dis=[]
    all_strdis=[]

    context_size = 2

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
        for pid ,sess in data[key].items():
            print("pid:",pid)
            context = []
            for turn in sess['dialogue']:
                if mode == 'User' and turn['speaker'] == '医生':
                    context.append(turn["speaker"] +"："+turn['sentence'])
                    continue
                elif mode == 'Doctor' and turn['speaker'] == '患者':
                    context.append(turn["speaker"] +"："+turn['sentence'])
                    continue
                sent = list(turn["speaker"] +"："+turn['sentence'])
                bio = ['O'] * 3 + turn['BIO_label'].split(' ')
                assert len(sent) == len(bio)
                assert turn["dialogue_act"] in tag2id
                if turn["dialogue_act"] not in ['Diagnose','Other']:
                    DA_act=turn["dialogue_act"].split('-')[0]
                    DA_slot=turn["dialogue_act"].split('-')[1]
                    DA_domain=""
                    DA_value=""
                else:
                    DA_act=turn["dialogue_act"]
                    DA_slot=""
                    DA_domain=""
                    DA_value=""
                # intents.append('+'.join([i['act_label'], j['domain'], j['slot'], j['value']]))
                golden = []
                intents = []
                tokens=[]
                bio_labels=[]
                intents.append('+'.join([DA_act, DA_domain, DA_slot, DA_value]))
                golden.append([DA_act, DA_domain, DA_slot, DA_value])
                for word ,label in zip(sent,bio):
                    token = tokenizer.tokenize(word)
                    if len(token) == 0:
                        token = ['[UNK]']
                        print("UNK: ", word)
                    if len(token) > 1:
                        print("token: ", token)
                    tokens.extend(token)
                    bio_labels.extend([label] + ["[PAD]"] * (len(token) - 1))

                assert len(tokens) == len(bio_labels)
                slot_DA=tag2das(tokens,bio_labels)
                golden.extend(slot_DA)
                processed_data[key].append([tokens, bio_labels, intents, golden, context[-context_size:]])
                # print([tokens, tags, intents, golden, context[-context_size:]])
                # input()

                all_intent += intents
                all_tag += bio_labels
                # all_domain+=domain
                # all_act+=act
                # for i in golden:
                #     if i not in all_golden:
                #         all_golden += [i.copy()]
                # for j in dis:
                #     '''j=  [
                #             "Inform",
                #             "饮食",
                #             "饮食名"
                #         ],'''
                #     strj=j[0]+"-"+j[1]+"-"+j[2]
                #     all_strdis.append(strj)
                #     if j not in all_dis:
                #         all_dis += [j.copy()]
                    

                context.append(turn["speaker"] +"："+turn['sentence'])

        all_intent = [x[0] for x in dict(Counter(all_intent)).items()]
        all_tag = [x[0] for x in dict(Counter(all_tag)).items()]
        # all_domain=[x[0] for x in dict(Counter(all_domain)).items()]
        # all_act=[x[0] for x in dict(Counter(all_act)).items()]
        # all_golden = [x[0] for x in dict(Counter(all_golden)).items()]
        # strdis_count=dict(sorted(dict(Counter(all_strdis)).items(),key=lambda x:x[1]))
        # print(strdis_count)
        


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
    # json.dump(all_domain, open(os.path.join(processed_data_dir, 'domain_vocab.json'), 'w', encoding='utf-8'), indent=2,
    #           ensure_ascii=False)
    # json.dump(all_act, open(os.path.join(processed_data_dir, 'act_vocab.json'), 'w', encoding='utf-8'), indent=2,
    #           ensure_ascii=False)
    # json.dump(all_golden, open(os.path.join(processed_data_dir, 'golden_vocab.json'), 'w', encoding='utf-8'), indent=2,
    #             ensure_ascii=False)
    # json.dump(all_dis, open(os.path.join(processed_data_dir, 'dis_vocab.json'), 'w', encoding='utf-8'), indent=2,
    #             ensure_ascii=False)
    # json.dump(strdis_count, open(os.path.join(processed_data_dir, 'strdis_count.json'), 'w', encoding='utf-8'), indent=2,
    #             ensure_ascii=False)

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
