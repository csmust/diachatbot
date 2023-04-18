import numpy as np
import torch
import random
from transformers import BertTokenizer
import math
from collections import Counter



class Dataloader:
    def __init__(self, intent_vocab, tag_vocab, domain_vocab, act_vocab, pretrained_weights):
        """
        :param intent_vocab: list of all intents
        :param tag_vocab: list of all tags
        :param pretrained_weights: which bert, e.g. 'bert-base-uncased'
        """
        self.intent_vocab = intent_vocab
        self.tag_vocab = tag_vocab
        self.domain_vocab = domain_vocab
        self.act_vocab = act_vocab

        self.intent_dim = len(intent_vocab)   # 58
        self.tag_dim = len(tag_vocab)   # 470
        self.domain_dim = len(domain_vocab)
        self.act_dim = len(act_vocab)

        # 每一条索引和对应的内容
        self.id2intent = dict([(i, x) for i, x in enumerate(intent_vocab)])
        self.intent2id = dict([(x, i) for i, x in enumerate(intent_vocab)])
        self.id2tag = dict([(i, x) for i, x in enumerate(tag_vocab)])
        self.tag2id = dict([(x, i) for i, x in enumerate(tag_vocab)])
        self.id2domain = dict([(i, x) for i, x in enumerate(domain_vocab)])
        self.domain2id = dict([(x, i) for i, x in enumerate(domain_vocab)])
        self.id2act = dict([(i, x) for i, x in enumerate(act_vocab)])
        self.act2id = dict([(x, i) for i, x in enumerate(act_vocab)])
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.data = {}
        # 初始化每个intent的权重均为1
        self.intent_weight = [1] * len(self.intent2id)
        self.domain_weight = [1] * len(self.domain2id)
        self.act_weight = [1] * len(self.act2id)

    def load_data(self, data, data_key, cut_sen_len, use_bert_tokenizer=True):
        """
        sample representation: [list of words, list of tags, list of intents, original dialog act]
        :param data_key: train/val/test
        :param data:
        :return:
        """
        self.data[data_key] = data
        max_sen_len, max_context_len = 0, 0
        sen_len = []
        context_len = []
        for d in self.data[data_key]:
            max_sen_len = max(max_sen_len, len(d[0]))
            sen_len.append(len(d[0]))
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"], context(list of str))
            if cut_sen_len > 0:
                d[0] = d[0][:cut_sen_len]
                d[1] = d[1][:cut_sen_len]
                d[4] = [' '.join(s.split()[:cut_sen_len]) for s in d[4]]

            d[4] = self.tokenizer.encode('[CLS] ' + ' [SEP] '.join(d[4]))
            max_context_len = max(max_context_len, len(d[4]))
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

            domain_seq = list(set([x[1] for x in d[3]]))
            d.append(sorted(self.seq_domain2id(domain_seq)))

            act_seq = list(set([x[0] for x in d[3]]))
            d.append(sorted(self.seq_act2id(act_seq)))

            # d = (tokens, tags, intents, da2triples(turn["dialog_act"]), context(token id), new2ori, new_word_seq, tag2id_seq, intent2id_seq,domain2id_seq,act2id_seq)
            # d     0       1      2           3                                4              5           6           7            8              9           10
            if data_key == 'train':
                for intent_id in d[8]:
                    self.intent_weight[intent_id] += 1
                for domain_id in d[9]:
                    self.domain_weight[domain_id] += 1
                for act_id in d[10]:
                    self.act_weight[act_id] += 1
        print(self.intent_weight)
        print(self.domain_weight)
        print(self.act_weight)
        if data_key == 'train':
            train_size = len(self.data['train'])
            l1,l2,l3=[],[],[]
            for intent, intent_id in self.intent2id.items():
                neg_pos = (train_size - self.intent_weight[intent_id]) / self.intent_weight[intent_id]
                l1.append(neg_pos)
                self.intent_weight[intent_id] = np.log10(neg_pos)
            print(l1)
            print(self.intent_weight)
            self.intent_weight = torch.tensor(self.intent_weight)


            for domain, domain_id in self.domain2id.items():
                neg_pos = (train_size - self.domain_weight[domain_id]) / self.domain_weight[domain_id]
                l2.append(neg_pos)
                self.domain_weight[domain_id] = np.log10(neg_pos)+1
            print(l2)
            print(self.domain_weight)
            self.domain_weight = torch.tensor(self.domain_weight)

            for act, act_id in self.act2id.items():
                neg_pos = (train_size - self.act_weight[act_id]) / self.act_weight[act_id]
                l3.append(neg_pos)
                self.act_weight[act_id] = np.log10(neg_pos)+1
            print(l3)
            print(self.act_weight)
            self.act_weight = torch.tensor(self.act_weight)

        print('max sen bert len', max_sen_len)
        print(sorted(Counter(sen_len).items()))
        print('max context bert len', max_context_len)
        print(sorted(Counter(context_len).items()))

        # done

    def bert_tokenize(self, word_seq, tag_seq):
        split_tokens = []
        new_tag_seq = []
        new2ori = {}
        basic_tokens = self.tokenizer.basic_tokenizer.tokenize(
            ' '.join(word_seq))
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

    def seq_domain2id(self, domains):
        return [self.domain2id[x] for x in domains if x in self.domain2id]

    def seq_id2domain(self, ids):
        return [self.id2domain[x] for x in ids]

    def seq_act2id(self, acts):
        return [self.act2id[x] for x in acts if x in self.act2id]

    def seq_id2domain(self, ids):
        return [self.id2act[x] for x in ids]

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
    def pad_batch(self, batch_data):
        batch_size = len(batch_data)
        # max_seq_len = max([len(x[-3]) for x in batch_data]) + 2
        max_seq_len = max([len(x[6]) for x in batch_data]) + 2
        word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        intent_tensor = torch.zeros((batch_size, self.intent_dim), dtype=torch.float)
        domain_tensor = torch.zeros((batch_size, self.domain_dim), dtype=torch.float)
        act_tensor = torch.zeros((batch_size, self.act_dim), dtype=torch.float)
        context_max_seq_len = max([len(x[4]) for x in batch_data])
        context_mask_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        context_seq_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        for i in range(batch_size):
            # words = batch_data[i][-3]
            # tags = batch_data[i][-2]
            # intents = batch_data[i][-1]
            words = batch_data[i][6]
            tags = batch_data[i][7]
            intents = batch_data[i][8]
            domains = batch_data[i][9]
            acts = batch_data[i][10]

            words = ['[CLS]'] + words + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(words)
            sen_len = len(words)
            word_seq_tensor[i, :sen_len] = torch.LongTensor([indexed_tokens])
            tag_seq_tensor[i, 1:sen_len - 1] = torch.LongTensor(tags)
            word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)
            tag_mask_tensor[i, 1:sen_len - 1] = torch.LongTensor([1] * (sen_len - 2))
            for j in intents:
                intent_tensor[i, j] = 1.
            for j in domains:
                domain_tensor[i, j] = 1.
            for j in acts:
                act_tensor[i, j] = 1.
            context_len = len(batch_data[i][4])
            context_seq_tensor[i, :context_len] = torch.LongTensor([batch_data[i][4]])
            context_mask_tensor[i, :context_len] = torch.LongTensor([1] * context_len)

        return word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor, domain_tensor, act_tensor

    def get_train_batch(self, batch_size):
        batch_data = random.choices(self.data['train'], k=batch_size)
        return self.pad_batch(batch_data)

    def yield_batches(self, batch_size, data_key):
        batch_num = math.ceil(len(self.data[data_key]) / batch_size)
        for i in range(batch_num):
            batch_data = self.data[data_key][i *
                                             batch_size:(i + 1) * batch_size]
            yield self.pad_batch(batch_data), batch_data, len(batch_data)
