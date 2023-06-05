import numpy as np
import torch
import random
from transformers import BertTokenizer
import math
from collections import Counter

class Dataloader:
    def __init__(self, intent_vocab, tag_vocab, pretrained_weights):
        """
        :param intent_vocab: list of all intents
        :param tag_vocab: list of all tags
        :param pretrained_weights: which bert, e.g. 'bert-base-uncased'
        """
        self.intent_vocab = intent_vocab
        self.tag_vocab = tag_vocab
        self.intent_dim = len(intent_vocab)   # 58
        self.tag_dim = len(tag_vocab)   # 470
        # 每一条索引和对应的内容
        self.id2intent = dict([(i, x) for i, x in enumerate(intent_vocab)])
        self.intent2id = dict([(x, i) for i, x in enumerate(intent_vocab)])
        self.id2tag = dict([(i, x) for i, x in enumerate(tag_vocab)])
        self.tag2id = dict([(x, i) for i, x in enumerate(tag_vocab)])
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.data = {}
        self.max_sen_len = 0  # max_seq_len =max_sen_len + 2
        self.max_context_len = 0  
        # 初始化每个intent的权重均为1
        self.intent_weight = [1] * len(self.intent2id)
    def load_data(self, data, data_key, cut_sen_len, use_bert_tokenizer=True):
        """
        sample representation: [list of words, list of tags, list of intents, original dialog act]
        :param data_key: train/val/test
        :param data:
        :return: 
        """
        self.data[data_key] = data
        # max_sen_len, max_context_len = 0, 0
        sen_len = []
        context_len = []
        for d in self.data[data_key]:
            self.max_sen_len = max(self.max_sen_len, len(d[0]))
            sen_len.append(len(d[0]))
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"], context(list of str))
            if cut_sen_len > 0:
                d[0] = d[0][:cut_sen_len]
                d[1] = d[1][:cut_sen_len]
                d[4] = [' '.join(s.split()[:cut_sen_len]) for s in d[4]]

            d[4] = self.tokenizer.encode('[CLS] ' + ' [SEP] '.join(d[4]))
            self.max_context_len = max(self.max_context_len, len(d[4]))
            context_len.append(len(d[4]))

            if use_bert_tokenizer:
                word_seq, tag_seq, new2ori = self.bert_tokenize(d[0], d[1])
            else:
                word_seq = d[0]
                tag_seq = d[1]
                new2ori = None
            d.append(new2ori)
            d.append(word_seq)

            d.append(self.seq_tag2id(tag_seq))
            d.append(self.seq_intent2id(d[2]))
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"]), context(token id), new2ori, new_word_seq, tag2id_seq, intent2id_seq)
            if data_key == 'train':
                for intent_id in d[-1]:
                    self.intent_weight[intent_id] += 1
        if data_key == 'train':
            train_size = len(self.data['train'])
            for intent, intent_id in self.intent2id.items():
                neg_pos = (train_size - self.intent_weight[intent_id]) / self.intent_weight[intent_id]
                self.intent_weight[intent_id] = np.log10(neg_pos)
            self.intent_weight = torch.tensor(self.intent_weight)
        print('max sen bert len', self.max_sen_len)
        print(sorted(Counter(sen_len).items()))
        print('max context bert len', self.max_context_len)
        print(sorted(Counter(context_len).items()))

    def bert_tokenize(self, word_seq, tag_seq):
        split_tokens = []
        new_tag_seq = []
        new2ori = {}
        basic_tokens = self.tokenizer.basic_tokenizer.tokenize(' '.join(word_seq))
        accum = ''
        i, j = 0, 0
        for i, token in enumerate(basic_tokens):
            if (accum + token).lower() == word_seq[j].lower():
                accum = ''
            else:
                accum += token
            for sub_token in self.tokenizer.wordpiece_tokenizer.tokenize(basic_tokens[i]):
                new2ori[len(new_tag_seq)] = j
                split_tokens.append(sub_token)
                new_tag_seq.append(tag_seq[j])
            if accum == '':
                j += 1
        return split_tokens, new_tag_seq, new2ori

    # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    def seq_tag2id(self, tags):
        return [self.tag2id[x] if x in self.tag2id else self.tag2id['O'] for x in tags]

    def seq_id2tag(self, ids):
        return [self.id2tag[x] for x in ids]

    # []
    def seq_intent2id(self, intents):
        return [self.intent2id[x] for x in intents if x in self.intent2id]

    def seq_id2intent(self, ids):
        return [self.id2intent[x] for x in ids]

    # [[
    # ['我', '吃', '莲', '藕', '排', '骨', '还', '喝', '汤', '，', '血', '糖', '升', '高', '，', '是', '不', '是', '不', '能', '喝', '呀', '？'],
    # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    # [],
    # {},
    # [101, 101, 102],
    # None,
    # ['我', '吃', '莲', '藕', '排', '骨', '还', '喝', '汤', '，', '血', '糖', '升', '高', '，', '是', '不', '是', '不', '能', '喝', '呀', '？'],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # []
    # ]]

    '''

    :
['这', '边', '为', '您', '推', '荐', '胡', '椒', '厨', '房', '(', '通', '州', '万', ...]
1:
['O', 'O', 'O', 'O', 'O', 'O', 'B+Recommend+餐馆+名称', 'I+Recommend+餐馆+名称', 'I+Recommend+餐馆+名称', 'I+Recommend+餐馆+名称', 'I+Recommend+餐馆+名称', 'I+Recommend+餐馆+名称', 'I+Recommend+餐馆+名称', 'I+Recommend+餐馆+名称', ...]
2:
[]
3:
[['Recommend', '餐馆', '名称', '呼伦贝尔草原蒙古包美食'], ['Recommend', '餐馆', '名称', '外婆家(万达店)'], ['Recommend', '餐馆', '名称', '胡椒厨房(通州万达广场店)']]
4:
[101, 101, 2644, 1962, 8024, 2769, 2682, 2823, 671, 2157, 6397, 1146, 711, 126, ...]
5:
None
6:
['这', '边', '为', '您', '推', '荐', '胡', '椒', '厨', '房', '(', '通', '州', '万', ...]
7:
[0, 0, 0, 0, 0, 0, 47, 48, 48, 48, 48, 48, 48, 48, ...]
8:
[]
len():
9
    '''
    """
    0:
['不', '用', '谢', '！']
1:
['O', 'O', 'O', 'O']
2:
['General+welcome+none+none']
3:
[['General', 'welcome', 'none', 'none']]
4:
[101, 101, 2769, 2682, 1461, 1373, 794, 7471, 7987, 2255, 3180, 3952, 1277, 1168, ...]
5:
None
6:
['不', '用', '谢', '！']
7:
[0, 0, 0, 0]
8:
[15]
len():
9
    """
    def pad_batch(self, batch_data):
        batch_size = len(batch_data)
        max_seq_len = max([len(x[-3]) for x in batch_data]) + 2
        word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        intent_tensor = torch.zeros((batch_size, self.intent_dim), dtype=torch.float)
        context_max_seq_len = max([len(x[-5]) for x in batch_data])
        context_mask_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        context_seq_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        for i in range(batch_size):
            words = batch_data[i][-3]
            tags = batch_data[i][-2]
            intents = batch_data[i][-1]
            words = ['[CLS]'] + words + ['[SEP]']  # 先在首位加标记
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(words)  #然后直接转成tokenid   pad是0
            sen_len = len(words)
            word_seq_tensor[i, :sen_len] = torch.LongTensor([indexed_tokens]) 
            tag_seq_tensor[i, 1:sen_len - 1] = torch.LongTensor(tags)  
            word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)   # SEP 和 CLS也是1
            tag_mask_tensor[i, 1:sen_len - 1] = torch.LongTensor([1] * (sen_len - 2))   # SEP 和 CLS是0
            for j in intents:
                intent_tensor[i, j] = 1.
            context_len = len(batch_data[i][-5])
            context_seq_tensor[i, :context_len] = torch.LongTensor([batch_data[i][-5]])
            context_mask_tensor[i, :context_len] = torch.LongTensor([1] * context_len)

        return word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor

    def get_train_batch(self, batch_size):
        batch_data = random.choices(self.data['train'], k=batch_size)
        return self.pad_batch(batch_data)

    def yield_batches(self, batch_size, data_key):
        batch_num = math.ceil(len(self.data[data_key]) / batch_size)
        for i in range(batch_num):
            batch_data = self.data[data_key][i * batch_size:(i + 1) * batch_size]
            yield self.pad_batch(batch_data), batch_data, len(batch_data)

