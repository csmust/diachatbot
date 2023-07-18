# -*- coding: utf-8 -*-#

import os
import numpy as np
from copy import deepcopy
from collections import Counter
from collections import OrderedDict
from ordered_set import OrderedSet

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class Alphabet(object):
    """
    Storage and serialization a set of elements.
    """

    def __init__(self, name, if_use_pad, if_use_unk):

        self.__name = name #   'word'  'slot'  'intent'
        self.__if_use_pad = if_use_pad
        self.__if_use_unk = if_use_unk

        self.__index2instance = OrderedSet()  # OrderedSet()保证了集合的有序性 OrderedSet(['[PAD]', '<UNK>', '医', '生'])
        self.__instance2index = OrderedDict()    # OrderedDict()保证了字典的有序性 OrderedDict([('[PAD]', 0), ('<UNK>', 1), ('医', 2), ('生', 3)])

        # Counter Object record the frequency
        # of element occurs in raw text.
        self.__counter = Counter()

        if if_use_pad:
            self.__sign_pad = "<PAD>"
            self.add_instance(self.__sign_pad)
        if if_use_unk:
            self.__sign_unk = "<UNK>"
            self.add_instance(self.__sign_unk)

    @property
    def name(self):
        return self.__name

    def add_instance(self, instance, multi_intent=False):
        """ Add instances to alphabet.
        从训练集中采词，向词表中添加实例
        1, We support any iterative data structure which
        contains elements of str type.

        2, We will count added instances that will influence
        the serialization of unknown instance.

        :param instance: is given instance or a list of it.
        """
        #这代码真漂亮，递归直到instance是字符串
        if isinstance(instance, (list, tuple)):  #isinstance()用来判断一个对象是否是一个已知的类型 list, tuple
            for element in instance:
                self.add_instance(element, multi_intent=multi_intent)
            return

        # We only support elements of str type.   只支持字符串，
        assert isinstance(instance, str)
        if multi_intent and '#' in instance:
            for element in instance.split('#'):
                self.add_instance(element, multi_intent=multi_intent)
            return
        # count the frequency of instances.
        self.__counter[instance] += 1

        if instance not in self.__index2instance:
            self.__instance2index[instance] = len(self.__index2instance)
            self.__index2instance.append(instance)

    def get_index(self, instance, multi_intent=False):
        """ Serialize given instance and return.
        多意图默认为false

        For unknown words, the return index of alphabet
        depends on variable self.__use_unk:

            1, If True, then return the index of "<UNK>";
            2, If False, then return the index of the
            element that hold max frequency in training data.

        :param instance: is given instance or a list of it.
        :return: is the serialization of query instance.
        """

        if isinstance(instance, (list, tuple)):
            return [self.get_index(elem, multi_intent=multi_intent) for elem in instance]

        assert isinstance(instance, str)
        if multi_intent and '#' in instance:
            return [self.get_index(element, multi_intent=multi_intent) for element in instance.split('#')]

        try:
            return self.__instance2index[instance]
        except KeyError:
            if self.__if_use_unk:    #自建词表的弊端，验证集'袪'不在训练集中出现，返回<UNK>的序号  '浸''熏' '绊''邦''卵''怒'
                return self.__instance2index[self.__sign_unk]
            else:
                max_freq_item = self.__counter.most_common(1)[0][0]
                return self.__instance2index[max_freq_item]

    def get_instance(self, index):
        """ Get corresponding instance of query index.

        if index is invalid, then throws exception.

        :param index: is query index, possibly iterable.
        :return: is corresponding instance.
        """

        if isinstance(index, list):
            return [self.get_instance(elem) for elem in index]

        return self.__index2instance[index]

    def save_content(self, dir_path):
        """ Save the content of alphabet to files.

        There are two kinds of saved files:
            1, The first is a list file, elements are
            sorted by the frequency of occurrence.

            2, The second is a dictionary file, elements
            are sorted by it serialized index.

        :param dir_path: is the directory path to save object.
        """

        # Check if dir_path exists.
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        list_path = os.path.join(dir_path, self.__name + "_list.txt")
        with open(list_path, 'w', encoding="utf8") as fw:
            for element, frequency in self.__counter.most_common():   #most_common()返回一个频率TopN列表。如果n没有被指定，则返回所有元素。 
                fw.write(element + '\t' + str(frequency) + '\n')

        dict_path = os.path.join(dir_path, self.__name + "_dict.txt")
        with open(dict_path, 'w', encoding="utf8") as fw:
            for index, element in enumerate(self.__index2instance):  #index =0   element ="<PAD>"
                fw.write(element + '\t' + str(index) + '\n')

    def __len__(self):
        return len(self.__index2instance)

    def __str__(self):
        return 'Alphabet {} contains about {} words: \n\t{}'.format(self.name, len(self), self.__index2instance)


class TorchDataset(Dataset):
    """
    Helper class implementing torch.utils.data.Dataset to
    instantiate DataLoader which deliveries data batch.
    """

    def __init__(self, text, slot, intent):
        self.__text = text
        self.__slot = slot
        self.__intent = intent

    def __getitem__(self, index):
        return self.__text[index], self.__slot[index], self.__intent[index]

    def __len__(self):
        # Pre-check to avoid bug.
        assert len(self.__text) == len(self.__slot)
        assert len(self.__text) == len(self.__intent)
        return len(self.__text)


class DatasetManager(object):

    def __init__(self, args):

        # Instantiate alphabet objects.
        # self.__word_alphabet = Alphabet('word', if_use_pad=True, if_use_unk=True)
        self.__slot_alphabet = Alphabet('slot', if_use_pad=False, if_use_unk=False)
        self.__intent_alphabet = Alphabet('intent', if_use_pad=False, if_use_unk=False)

        # Record the raw text of dataset.   原始文本数据
        self.__text_word_data = {}
        self.__text_slot_data = {}
        self.__text_intent_data = {}

        # Record the serialization of dataset.  数字序列化的数据
        self.__digit_word_data = {} #字典
        self.__digit_slot_data = {}
        self.__digit_intent_data = {}

        self.__args = args
        self.tokenizer = args.tokenizer 

    @property
    def test_sentence(self):
        return deepcopy(self.__text_word_data['test'])

    # @property
    # def word_alphabet(self):
    #     return deepcopy(self.__word_alphabet)

    @property
    def slot_alphabet(self):
        return deepcopy(self.__slot_alphabet)

    @property
    def intent_alphabet(self):
        return deepcopy(self.__intent_alphabet)

    @property
    def num_epoch(self):
        return self.__args.num_epoch

    @property
    def batch_size(self):
        return self.__args.batch_size

    @property
    def learning_rate(self):
        return self.__args.learning_rate

    @property
    def l2_penalty(self):
        return self.__args.l2_penalty

    @property
    def save_dir(self):
        return self.__args.save_dir

    @property
    def slot_forcing_rate(self):
        return self.__args.slot_forcing_rate
    
    @property
    def use_bert(self):
        return self.__args.use_bert

    def show_summary(self):
        """
        :return: show summary of dataset, training parameters.
        """

        print("Training parameters are listed as follows:\n")

        print('\tnumber of train sample:                    {};'.format(len(self.__text_word_data['train'])))
        print('\tnumber of dev sample:                      {};'.format(len(self.__text_word_data['dev'])))
        print('\tnumber of test sample:                     {};'.format(len(self.__text_word_data['test'])))
        print('\tnumber of epoch:						    {};'.format(self.num_epoch))
        print('\tbatch size:							    {};'.format(self.batch_size))
        print('\tlearning rate:							    {};'.format(self.learning_rate))
        print('\trandom seed:							    {};'.format(self.__args.random_state))
        print('\trate of l2 penalty:					    {};'.format(self.l2_penalty))
        print('\trate of dropout in network:                {};'.format(self.__args.dropout_rate))
        print('\tteacher forcing rate(slot)		    		{};'.format(self.slot_forcing_rate))

        print("\nEnd of parameters show. Save dir: {}.\n\n".format(self.save_dir))

    def quick_build(self):
        """
        Convenient function to instantiate a dataset object.
        """

        train_path = os.path.join(self.__args.data_dir, 'train.txt')
        dev_path = os.path.join(self.__args.data_dir, 'dev.txt')
        test_path = os.path.join(self.__args.data_dir, 'test.txt')

        self.add_file(train_path, 'train', if_train_file=True)  # 训练集
        self.add_file(dev_path, 'dev', if_train_file=False)
        self.add_file(test_path, 'test', if_train_file=False)

        # Check if save path exists.
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        alphabet_dir = os.path.join(self.__args.save_dir, "alphabet")
        # self.__word_alphabet.save_content(alphabet_dir)
        self.__slot_alphabet.save_content(alphabet_dir)
        self.__intent_alphabet.save_content(alphabet_dir)

    def get_dataset(self, data_name, is_digital):
        """ Get dataset of given unique name.

        :param data_name: is name of stored dataset.
        :param is_digital: make sure if want serialized data.
        :return: the required dataset.
        """

        if is_digital:
            return self.__digit_word_data[data_name], \
                   self.__digit_slot_data[data_name], \
                   self.__digit_intent_data[data_name]
        else:
            return self.__text_word_data[data_name], \
                   self.__text_slot_data[data_name], \
                   self.__text_intent_data[data_name]

    def bert_get_text(self, text,digital=True):
        """
        返回序列化的数据[[2, 3, 4, 5, 6], [2, 3, 4, 7, 8], [9, 10, 4, 11, 12, 13, 13, 14, 15, ...], [2, 3, 4, 17, 5, 18, 19, 20, 21, ...], [9, 10, 4, 35, 14, 15, 19, 36, 37, ...], [2, 3, 4, 42, 43, 44, 19, 45, 46, ...], [2, 3, 4, 55, 39, 40, 41, 56, 35, ...], [2, 3, 4, 11, 12, 24, 25, 59, 60, ...], [9, 10, 4, 67, 48, 68, 69, 70, 71, ...], [2, 3, 4, 73, 7, 11, 12, 24, 25, ...], [2, 3, 4, 49, 44, 76, 77, 78, 79, ...], [9, 10, 4, 86, 87, 88, 89, 62, 90, ...], [2, 3, 4, 94, 93, 26, 27, 95, 96, ...], [9, 10, 4, 91, 97, 96, 98], ...]
        并前后添加 101 102
        """
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        # pad_token_id = self.tokenizer.pad_token_id
        if digital:
            return [[cls_token_id] + self.tokenizer.convert_tokens_to_ids(t) + [sep_token_id] for t in text]
        else:
            return [ self.tokenizer.convert_tokens_to_ids(t)  for t in text]
    
        pass

    def add_file(self, file_path, data_name, if_train_file):  #data_name='train' ，'dev'，'test'  
        text, slot, intent = self.__read_file(file_path)

        if if_train_file: # 只从训练集建立词表 槽表 意图表
            # self.__word_alphabet.add_instance(text)  #text[['医', '生', '：', '你', '好'], ['医', '生', '：', '在', '吗'], ['患', '者', '：', '这', '个', '粑', '粑', '正', '常', ...], ['医', '生', '：', '从', '你', '发', '的', '图', '片', ...], ['患', '者', '：', '不', '正', '常', '的', '话', '和', ...], ['医', '生', '：', '最', '主', '要', '的', '问', '题', ...], ['医', '生', '：', '跟', '奶', '粉', '关', '系', '不', ...], ['医', '生', '：', '这', '个', '孩', '子', '使', '用', ...], ['患', '者', '：', '那', '是', '怎', '么', '回', '事', ...], ['医', '生', '：', '现', '在', '这', '个', '孩', '子', ...], ['医', '生', '：', '需', '要', '考', '虑', '消', '化', ...], ['患', '者', '：', '刚', '开', '始', '没', '多', '久', ...], ['医', '生', '：', '每', '天', '大', '便', '几', '次', ...], ['患', '者', '：', '四', '五', '次', '吧'], ...]# 将文本转化为数字序列 BERT TODO
            self.__slot_alphabet.add_instance(slot) #slot[['O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Symptom', 'I-Symptom', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O'], ...]
            self.__intent_alphabet.add_instance(intent, multi_intent=True) #intent[['Other'], ['Other'], ['Request-Symptom'], ['Inform-Symptom'], ['Request-Etiology'], ['Inform-Medical_Advice'], ['Inform-Etiology'], ['Request-Basic_Information'], ['Request-Etiology'], ['Request-Basic_Information'], ['Inform-Etiology'], ['Inform-Basic_Information'], ['Request-Symptom'], ['Inform-Symptom'], ...]

        # Record the raw text of dataset.
        self.__text_word_data[data_name] = text
        self.__text_slot_data[data_name] = slot
        self.__text_intent_data[data_name] = intent

        # Serialize raw text and stored it.
        #self.___digit_word_data[data_name] = self.__word_alphabet.get_index(text) # 所有数据集都这样，将文本转化为数字序列 "train" : [[2, 3, 4, 5, 6], [2, 3, 4, 7, 8], [9, 10, 4, 11, 12, 13, 13, 14, 15, ...], [2, 3, 4, 17, 5, 18, 19, 20, 21, ...], [9, 10, 4, 35, 14, 15, 19, 36, 37, ...], [2, 3, 4, 42, 43, 44, 19, 45, 46, ...], [2, 3, 4, 55, 39, 40, 41, 56, 35, ...], [2, 3, 4, 11, 12, 24, 25, 59, 60, ...], [9, 10, 4, 67, 48, 68, 69, 70, 71, ...], [2, 3, 4, 73, 7, 11, 12, 24, 25, ...], [2, 3, 4, 49, 44, 76, 77, 78, 79, ...], [9, 10, 4, 86, 87, 88, 89, 62, 90, ...], [2, 3, 4, 94, 93, 26, 27, 95, 96, ...], [9, 10, 4, 91, 97, 96, 98], ...]
        self.__digit_word_data[data_name]= self.bert_get_text(text) # BERT TODO
        
        
        if if_train_file:  # 只从训练集建立槽表的数字化 意图表的数字化
            self.__digit_slot_data[data_name] =[ [self.__slot_alphabet.get_index("O")]+x+[self.__slot_alphabet.get_index("O")] for x in self.__slot_alphabet.get_index(slot)]
            self.__digit_intent_data[data_name] = self.__intent_alphabet.get_index(intent, multi_intent=True)

    @staticmethod
    def __read_file(file_path):
        """ Read data file of given path.

        :param file_path: path of data file.
        :return: list of sentence, list of slot and list of intent.
        """

        texts, slots, intents = [], [], []
        text, slot = [], []

        with open(file_path, 'r', encoding="utf8") as fr:
            for line in fr.readlines():
                items = line.strip().split()  #'医 O\n'   items='医 O\n'

                if len(items) == 1:  #  意图
                    texts.append(text)
                    slots.append(slot)
                    if "/" not in items[0]:
                        intents.append(items)
                    else:
                        new = items[0].split("/")
                        intents.append([new[1]])

                    # clear buffer lists.
                    text, slot = [], []

                elif len(items) == 2:   # 单个字 和 对应的槽
                    text.append(items[0].strip())
                    slot.append(items[1].strip())

        return texts, slots, intents

    def batch_delivery(self, data_name, batch_size=None, is_digital=True, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size

        if is_digital:
            text = self.__digit_word_data[data_name]#[[2, 3, 4, 5, 6], [2, 3, 4, 7, 8], [9, 10, 4, 11, 12, 13, 13, 14, 15, ...], [2, 3, 4, 17, 5, 18, 19, 20, 21, ...], [9, 10, 4, 35, 14, 15, 19, 36, 37, ...], [2, 3, 4, 42, 43, 44, 19, 45, 46, ...], [2, 3, 4, 55, 39, 40, 41, 56, 35, ...], [2, 3, 4, 11, 12, 24, 25, 59, 60, ...], [9, 10, 4, 67, 48, 68, 69, 70, 71, ...], [2, 3, 4, 73, 7, 11, 12, 24, 25, ...], [2, 3, 4, 49, 44, 76, 77, 78, 79, ...], [9, 10, 4, 86, 87, 88, 89, 62, 90, ...], [2, 3, 4, 94, 93, 26, 27, 95, 96, ...], [9, 10, 4, 91, 97, 96, 98], ...]
            slot = self.__digit_slot_data[data_name]#[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 1, 2, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0], ...]
            assert len(text) == len(slot)
            for i in range(len(text)):
                assert len(text[i]) == len(slot[i])
            intent = self.__digit_intent_data[data_name]#[[0], [0], [1], [2], [3], [4], [5], [6], [3], [6], [5], [7], [1], [2], ...]
        else:
            text = self.__text_word_data[data_name] #[['医', '生', '：', '你', '好', '，', '咳', '嗽', '是', ...], ['医', '生', '：', '咳', '嗽', '有', '几', '天', '了', ...], ['医', '生', '：', '有', '发', '热', '过', '吗', '？'], ['患', '者', '：', '有', '三', '天'], ['患', '者', '：', '没', '发', '烧', '，', '也', '没', ...], ['医', '生', '：', '以', '前', '有', '气', '喘', '吗', ...], ['医', '生', '：', '有', '没', '什', '么', '过', '敏', ...], ['患', '者', '：', '没', '有'], ['医', '生', '：', '大', '便', '怎', '么', '样', '？', ...], ['患', '者', '：', '大', '便', '经', '常', '干', "'", ...], ['医', '生', '：', '可', '能', '有', '点', '积', '食'], ['患', '者', '：', '那', '该', '总', '么', '办'], ['医', '生', '：', '磨', '牙', '，', '晚', '上', '翻', ...], ['医', '生', '：', '现', '在', '可', '以', '吃', '点', ...], ...]
            slot = self.__text_slot_data[data_name]#[['O', 'O', 'O', 'O', 'O', 'O', 'B-Symptom', 'I-Symptom', 'O', ...], ['O', 'O', 'O', 'B-Symptom', 'I-Symptom', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'B-Symptom', 'I-Symptom', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'B-Symptom', 'I-Symptom', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'B-Symptom', 'I-Symptom', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Symptom', 'I-Symptom', ...], ['O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Symptom', 'I-Symptom'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'B-Symptom', 'I-Symptom', 'O', 'O', 'O', 'O', ...], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...], ...]
            intent = self.__text_intent_data[data_name]#[['Request-Symptom'], ['Request-Symptom'], ['Request-Symptom'], ['Inform-Symptom'], ['Inform-Symptom'], ['Request-Symptom'], ['Request-Symptom'], ['Inform-Symptom'], ['Request-Symptom'], ['Inform-Symptom'], ['Inform-Symptom'], ['Request-Medical_Advice'], ['Inform-Etiology'], ['Inform-Drug_Recommendation'], ...]
        
        dataset = TorchDataset(text, slot, intent)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.__collate_fn)  #__collate_fn什么作用？答：将一个batch的数据进一步处理返回real_batch

    @staticmethod
    def add_padding(texts, items=None, digital=True): #items= [(slot_batch, True), (intent_batch, False)]
        len_list = [len(text) for text in texts]  #batch_size中每个序列句子的长度  [9, 9, 19, 13, 8, 15, 15, 16, 14, 11, 14, 8, 24, 11, ...]
        if digital:
            max_len = max(len_list)  #其中最大
        else:
            max_len = max(len_list) + 2
        # Get sorted index of len_list.
        sorted_index = np.argsort(len_list)[::-1]#返回排序后的元素的索引 array([12,  2,  7,  6,  5, 10,  8,  3, 14, 13,  9, 15,  1,  0, 11,  4],dtype=int64)

        trans_texts, seq_lens, trans_items = [], [], None
        if items is not None:
            trans_items = [[] for _ in range(0, len(items))]  #[[], []]

        for index in sorted_index:  # index从最长的开始 array([12,  2,  7,  6,  5, 10,  8,  3, 14, 13,  9, 15,  1,  0, 11,  4],dtype=int64)·    
            
            trans_texts.append(deepcopy(texts[index]))
            if digital:
                trans_texts[-1].extend([0] * (max_len - len_list[index]))
                seq_lens.append(deepcopy(len_list[index]))
            else:
                trans_texts[-1]=["CLS"]+trans_texts[-1]+["SEP"]
                trans_texts[-1].extend(['[PAD]'] * (max_len- 2 - len_list[index]))
                seq_lens.append(deepcopy(len_list[index])+2)
                # assert len(trans_items[0])==seq_lens[0]
            # This required specific if padding after sorting.
            if items is not None:
                for item, (o_item, required) in zip(trans_items, items):
                    item.append(deepcopy(o_item[index]))
                    if required:
                        if digital:
                            item[-1].extend([0] * (max_len - len_list[index]))
                        else:
                            item[-1]=["CLS"]+item[-1]+["SEP"]
                            item[-1].extend(['[PAD]'] * (max_len-2 - len_list[index]))
        
        # torch.LongTensor(trans_items[0]) 
        if items is not None:  #trans_items=[[ [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 6, 0, 0, ...], [...], [...], [...], [...], [...], [...], [...], [...], ...[...] ],      [[11], [...], [...], [...], [...], [...], [...], [...], [...], ...]]
            return trans_texts, trans_items, seq_lens    #trans_texts(补0，按照原始长度排序，原始最长的在前面)=[[2, 3, 4, 131, 166, 684, 670, 38, 227, ...], [9, 10, 4, 1159, 1160, 129, 12, 42, 1209, ...], [2, 3, 4, 131, 166, 19, 23, 43, 44, ...], [9, 10, 4, 188, 69, 2048, 841, 421, 896, ...], [9, 10, 4, 26, 290, 49, 44, 1024, 95, ...], [9, 10, 4, 47, 367, 66, 114, 616, 491, ...], [2, 3, 4, 131, 166, 129, 186, 186, 615, ...], [2, 3, 4, 186, 186, 229, 230, 38, 39, ...], [9, 10, 4, 67, 49, 44, 38, 188, 69, ...], [9, 10, 4, 310, 23, 67, 103, 104, 48, ...], [9, 10, 4, 165, 135, 25, 122, 44, 68, ...], [9, 10, 4, 67, 48, 623, 1127, 241, 372, ...], [2, 3, 4, 114, 389, 6, 6, 404, 405, ...], [9, 10, 4, 28, 262, 23, 28, 157, 208, ...], ...]   seq_lens= [24, 19, 16, 15, 15, 14, 14, 13, 11, 11, 11, 10, 9, 9, ...]
        else:
            return trans_texts, seq_lens

    @staticmethod
    def __collate_fn(batch):
        """#根据训练集  digital是否有所不同，序列化还是文字
        helper function to instantiate a DataLoader Object. #在取出一个batch的数据后，对batch中的数据进行处理，将一个batch的数据进一步处理返回real_batch
        """#batch[0]=([9, 10, 4, 28, 262, 23, 28, 157, 208], [0, 0, 0, 0, 1, 0, 0, 0, 1], [2])
        #batch结构为[([...], [...], [...]), ([...], [...], [...]), ([...], [...], [...]), ([...], [...], [...]), ([...], [...], [...]), ([...], [...], [...]), ([...], [...], [...]), ([...], [...], [...]), ([...], [...], [...]), ([...], [...], [...]), ([...], [...], [...]), ([...], [...], [...]), ([...], [...], [...]), ([...], [...], [...]), ...]
        n_entity = len(batch[0])  #batch[0][0]是text的序列化id，batch[0][1]是slot的序列化id，batch[0][2]是intent的序列化id  n_entity=3
        modified_batch = [[] for _ in range(0, n_entity)]#[[], [], []]

        for idx in range(0, len(batch)):  #len(batch)=16
            for jdx in range(0, n_entity):
                modified_batch[jdx].append(batch[idx][jdx])

        return modified_batch  #[[[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...]]
'''                             #len(modified_batch)=3  modified_batch[0]=[[9, 10, 4, 28, 262, 23, 28, 157, 208], [2, 3, 4, 114, 389, 6, 6, 404, 405], [9, 10, 4, 1159, 1160, 129, 12, 42, 1209, ...], [2, 3, 4, 186, 186, 229, 230, 38, 39, ...], [9, 10, 4, 54, 180, 463, 309, 8], [9, 10, 4, 26, 290, 49, 44, 1024, 95, ...], [9, 10, 4, 188, 69, 2048, 841, 421, 896, ...], [2, 3, 4, 131, 166, 19, 23, 43, 44, ...], [2, 3, 4, 131, 166, 129, 186, 186, 615, ...], [9, 10, 4, 165, 135, 25, 122, 44, 68, ...], [9, 10, 4, 47, 367, 66, 114, 616, 491, ...], [9, 10, 4, 89, 28, 367, 368, 369], [2, 3, 4, 131, 166, 684, 670, 38, 227, ...], [9, 10, 4, 310, 23, 67, 103, 104, 48, ...], ...]
batch[0][0]                      # modified_batch[1]=[[0, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 3, 4, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 1, 2, 2, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 1, ...], [0, 0, 0, 0, 0, 1, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], [0, 0, 0, 0, 0, 0, 0, 0, 0, ...], ...]
0:                                 # modified_batch[2]=[[2], [9], [14], [1], [10], [10], [12], [11], [9], [10], [8], [2], [11], [0], ...]
[9, 10, 4, 28, 262, 23, 28, 157, 208]
1:
[0, 0, 0, 0, 1, 0, 0, 0, 1]
2:
[2]
'''