import os
import zipfile
import json
import torch

from dataloader import Dataloader
# from convlab2.nlu.jointBERT.diachat.jointBERT import JointBERT
from convlab2.nlu.nlu import NLU
from jointBERT_CRF import JointBERT_CRF
from postprocess import recover_intent
from preprocess import preprocess


class BERTNLU(NLU):
    def __init__(self, mode='All', config_file='all.json'):
        assert mode == 'All' or mode == 'User' or mode == 'Doctor'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'config/{}'.format(config_file))
        config = json.load(open(config_file,encoding='utf-8'))
        DEVICE = config['DEVICE']

        data_dir = os.path.join(current_dir,config['data_dir'])
        output_dir = os.path.join(current_dir,config['output_dir'])

        if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
            preprocess(mode)

        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json'), encoding='utf-8'))
        tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json'), encoding='utf-8'))
        dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])

        print('intent num:', len(intent_vocab))
        print('tag num:', len(tag_vocab))
        best_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        print('Load from', best_model_path)
        model = JointBERT_CRF(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim ,dataloader.max_sen_len,dataloader.max_context_len )
        # 对训练好的模型进行重新加载
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
        model.to(DEVICE)
        # 主要是针对model 在训练时和评价时不同的 Batch Normalization 和 Dropout 方法模式。
        model.eval()

        self.model = model
        self.dataloader = dataloader
        print("BERTNLU loaded")

    # utterance 是当前要理解的语句，context是上下文，以列表的形式传入
    def predict(self, utterance, context=list()):
        # 通过tokenizer 进行分字：['我', '吃', '莲', '藕', '排', '骨', '还', '喝', '汤', '，', '血', '糖', '升', '高', '，', '是', '不', '是', '不', '能', '喝', '呀', '？']
        ori_word_seq = self.dataloader.tokenizer.tokenize(utterance)
        # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        ori_tag_seq = ['O'] * len(ori_word_seq)
        # [101, 101, 102]
        context_seq = self.dataloader.tokenizer.encode('[CLS] ' + ' [SEP] '.join(context[-3:]))
        intents = []
        da = {}

        word_seq, tag_seq, new2ori = ori_word_seq, ori_tag_seq, None
        batch_data = [[ori_word_seq, ori_tag_seq, intents, da, context_seq,
                       new2ori, word_seq, self.dataloader.seq_tag2id(tag_seq), self.dataloader.seq_intent2id(intents)]]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to(self.model.device) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        tag_seq_id, intent_logits = self.model.forward(word_seq_tensor, word_mask_tensor,
                                                        context_seq_tensor=context_seq_tensor,
                                                        context_mask_tensor=context_mask_tensor)
        intent = recover_intent(self.dataloader, intent_logits[0], tag_seq_id[0], tag_mask_tensor[0],
                                batch_data[0][0], batch_data[0][-4])
        return intent



if __name__ == '__main__':
    nlu = BERTNLU(mode='All', config_file='all_context.json',
                  )
    # print(nlu.predict("我吃莲藕排骨还喝汤，血糖升高，是不是不能喝呀？"))
    # print(nlu.predict("血糖又高了，是不是我又吃多了"))
    # print(nlu.predict("我有凌晨三点测过也是7点多到8点的样子"))
    # print(nlu.predict("这个就是晨起反应"))
    # print(nlu.predict("这个就是糖尿病"))
    # print(nlu.predict("吃维生素B1B12维生素C会升血糖吗"))
    # print(nlu.predict("今年21岁了"))

    # print(nlu.predict("30岁"))
    # print(nlu.predict("7"))
    # print(nlu.predict("今年7岁了"))
    # print(nlu.predict("7",["你空腹血糖多少？"]))
    # print(nlu.predict("7",["多大了？"]))
    # # print(nlu.predict("180CM",["你身高多少？"]))
    # print(nlu.predict("180"))
    # print(nlu.predict("180",["你身高多少？"]))
    # print(nlu.predict("180CM",["你身高多少厘米？"]))
    # print(nlu.predict('这个考虑你喝的有点多',["我吃莲藕排骨还喝汤，血糖升高，是不是不能喝呀？"]))
    print(nlu.predict('如果您刚刚吃了好几个无糖沙琪码，建议您注意控制饮食量，因为即使是无糖食品也可能含有高热量和高脂肪。建议您适量摄入低糖、低脂的食品，同时按照医生的建议进行药物治疗和控制血糖。'))
    
    '''
        输出形式为
    [['Inform', '饮食', '饮食名', '莲藕排骨'], ['AskForSure', '行为', '行为名', '喝'], ['Inform', '饮食', '饮食名', '汤'], ['Inform', '问题', '血糖值', '升高']]
    [['Inform', '问题', '血糖值', '又'], ['Inform', '问题', '血糖值', '高'], ['AskForSure', '饮食', '饮食量', '多']]
    [['Inform', '问题', '时间', '凌晨三点'], ['Inform', '问题', '血糖值', '7点多到8点']]
    '''


