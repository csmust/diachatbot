from distutils.command.config import config
from multiprocessing import reduction
import torch
from torch import nn
from transformers import BertModel
from torchcrf import CRF
from model_config import *
import copy


class JointBERT_CRF(nn.Module):
    def __init__(self, model_config, device, slot_dim, intent_dim, intent_weight=None):
        super().__init__()
        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.device = device
        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.]*intent_dim)
        print(model_config['pretrained_weights'])
        self.bert = BertModel.from_pretrained(model_config['pretrained_weights'])
        print(self.bert.config)
        self.dropout = nn.Dropout(model_config['dropout'])
        self.context = model_config['context']
        self.finetune = model_config['finetune']
        self.context_grad = model_config['context_grad']
        self.hidden_units = model_config['hidden_units']
        if self.hidden_units > 0:
            if self.context:
                self.intent_classifier = nn.Linear(
                    self.hidden_units, self.intent_num_labels)
                self.slot_classifier = nn.Linear(
                    self.hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(
                    2 * self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden = nn.Linear(
                    2 * self.bert.config.hidden_size, self.hidden_units)
            else:
                self.intent_classifier = nn.Linear(
                    self.hidden_units, self.intent_num_labels)
                self.slot_classifier = nn.Linear(
                    self.hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(
                    self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden = nn.Linear(
                    self.bert.config.hidden_size, self.hidden_units)
            nn.init.xavier_uniform_(self.intent_hidden.weight)
            nn.init.xavier_uniform_(self.slot_hidden.weight)
        else:
            if self.context:
                self.intent_classifier = nn.Linear(
                    2 * self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(
                    2 * self.bert.config.hidden_size, self.slot_num_labels)
            else:
                self.intent_classifier = nn.Linear(
                    self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(
                    self.bert.config.hidden_size, self.slot_num_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier.weight)

        # if LSTM_UNITS>0:
        #     self.lstm=nn.LSTM()
        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.intent_weight)

        if LAST_ADD_CRF:
            print("----BERT+CRF------")
            print("----请注意核对后处理.py 以及 train.py的model.forward()使用-------")
            self.crf = CRF(self.slot_num_labels, batch_first=True)
            # self.slot_loss_fct = self.crf.forward
        else:
            print("请使用CRF")
            exit()

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,  # 他们四个都是一样的，比如torch.Size([8, 51])
                intent_tensor=None, context_seq_tensor=None, context_mask_tensor=None):  # torch.Size([8, 58])  torch.Size([8, 60])  torch.Size([8, 60])
        '''
        return tag_seq_id, intent_logits, (crf_slot_loss), (intent_loss),
        '''
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(input_ids=word_seq_tensor,
                                    attention_mask=word_mask_tensor)
        else:
            outputs = self.bert(input_ids=word_seq_tensor,
                                attention_mask=word_mask_tensor)

        sequence_output = outputs[0]  # [8 ，51， 768]
        pooled_output = outputs[1]  # [8,768]

        if self.context and (context_seq_tensor is not None):
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.bert(
                        input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            else:
                context_output = self.bert(
                    input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]  # 取【1位置】[8,768]
            # print("context_output= ",context_output)
            # exit()
            sequence_output = torch.cat(
                # context_output.unsqueeze(1) torch.Size([8, 1, 768]) # .repeat(通道的重复倍数1, 行的重复倍数sequence_output.size(1), 列的重复倍数1) 后torch.Size([8, 51, 768])
                [context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),  # unsqueeze()函数起升维的作用
                 sequence_output], dim=-1)     # torch.Size([8, 51, 1536])
            pooled_output = torch.cat(
                [context_output, pooled_output], dim=-1)   # [8,1536]

        if self.hidden_units > 0:
            sequence_output = nn.functional.relu(self.slot_hidden(
                self.dropout(sequence_output)))  # [8, 51, 1536]
            pooled_output = nn.functional.relu(
                self.intent_hidden(self.dropout(pooled_output)))  # [8,1536]

        sequence_output = self.dropout(sequence_output)    # [8, 51, 1536]
        """
        crf 尝试：
            #val时:
             # tag_logits = [sequence_length, tag_dim]
             # intent_logits = [intent_dim]
             # tag_mask_tensor = [sequence_length]
        """
        if LAST_ADD_CRF:
            '''
            如果BERT+CRF
            '''
            # # 1、计算slot_logits后处理取得tag_seq或者此处直接返回tag_seq
            # slot_logits = self.slot_classifier(sequence_output)  # [8,51,470]
            # crf_pred_mask = copy.deepcopy(tag_mask_tensor == 1)
            # # print(crf_pred_mask)
            # # TODO 处理方式1：mask首位置CLS的mask 置1  否则会报错，后处理忽略掉CLS，
            # # 首位置CLS的mask是0 ，但这样会报错，
            # crf_pred_mask[:, 0] = True
            # # print(crf_pred_mask)
            # crf_seq = self.crf.decode(slot_logits, crf_pred_mask)
            # print(crf_seq)
            # outputs = (crf_seq,)

            # 1、计算slot_logits后处理取得tag_seq或者此处直接返回tag_seq
            slot_logits = self.slot_classifier(sequence_output)[:,1:,:]  # [8,51,470]
            crf_pred_mask = copy.deepcopy(tag_mask_tensor == 1)[:,1:]
            # print(crf_pred_mask)
            # TODO 处理方式2：截掉CLS位置从下个位置开始
            # 首位置CLS的mask是0 ，但这样会报错，

            crf_pred_mask[:, 0] = True
            # print(crf_pred_mask)
            crf_seq = self.crf.decode(slot_logits, crf_pred_mask)
            # print(crf_seq)
            outputs = (crf_seq,)

            # 2、计算intent_logits 后处理取得intent
            pooled_output = self.dropout(pooled_output)
            intent_logits = self.intent_classifier(pooled_output)
            outputs = outputs + (intent_logits,)

            # 3、计算slot损失
            if tag_seq_tensor is not None:
                crf_slot_loss = (-1)*self.crf(slot_logits,
                                              tag_seq_tensor[:,1:], crf_pred_mask,reduction='token_mean')
                outputs = outputs + (crf_slot_loss,)

            # 4、计算intent损失
            if intent_tensor is not None:
                intent_loss = self.intent_loss_fct(
                    intent_logits, intent_tensor)
                outputs = outputs + (intent_loss,)
        else:
            # 如果不接CRF
            print("请先使用接CRF，LAST_ADD_CRF=True")
            exit(0)

        # print(outputs)
        # tag_seq_id, intent_logits, (crf_slot_loss), (intent_loss),
        return outputs


if __name__ == "__main__":
    import json
    import os
    # print(os.path.abspath(__file__))
    # print(os.getcwd())
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(cur_dir, 'config/all_context_108.json')
    # print(config_file)
    conf = json.load(open(config_file))
    model_conf = conf["model"]
    # print(model_conf)
    device = "cuda:1"
    model = JointBERT_CRF(model_conf, device, 470, 58)
    # TODO crf这个包 不能以0打头
    tag_mask_tensor = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 0]])
    tag_seq_tensor = torch.tensor([[0,   0,   0,   0,   8,   9,   9,   9,   0,   0,   0,  10,  11,   8,
                                    9,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0],
                                  [0,   0,   0, 150, 151, 151,   0,   0, 150, 151,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                   0,   0,   0,   0,   0,   0,   0,   0,   0],
                                   [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  74,  75,  75,
                                    75,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0],
                                   [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 144, 145,   0,
                                    144, 145, 145, 145, 145,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0],
                                   [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0],
                                   [0, 304, 305,   0,   0,   0,   0,   0,   0,   0, 158, 159, 159, 159,
                                    0,   0,  84,   0,  16,  17,   1,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0],
                                   [0,  78,  79,   0,   0,   0, 163, 208, 264, 298, 298, 298, 298, 298,
                                    298, 298, 298,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0],
                                   [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                                    0,   0,   0,   0,   0,   0,   0,   0,   0]])
    word_mask_tensor = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1]])
    word_seq_tensor = torch.tensor([[101, 2456, 6379, 3221, 2208, 7030, 1914, 7623, 8024, 1377,  809,  678,
                                     1286, 1217,  671, 7623,  102,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0],
                                   [101, 2644, 3300, 6117, 1327, 7770, 2772, 5442, 6577, 6117, 1408, 8043,
                                    102,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                    0,    0,    0],
                                    [101, 1963, 3362, 2971, 1169, 6821, 3416, 1962,  117, 1377,  809, 2714,
                                     2714, 1121, 7030,  102,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0],
                                    [101, 1008,  872, 6821, 3416, 3683, 6772, 4937, 2137, 4638, 8024, 4958,
                                     5592, 1469, 3241, 7623,  123, 2207, 3198, 2218, 1377,  809, 4638,  511,
                                     102,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0],
                                    [101, 2769, 6230, 2533, 2802, 5536, 2270, 5162,  738,  679,  671, 2137,
                                     4638, 5543, 4937, 2137,  102,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0],
                                    [101, 1217, 7623, 1391, 1567, 3683, 6772, 1962, 8024, 2769, 1282,  671,
                                     4157, 1288, 1391, 4638, 7649, 8024, 4385, 1762, 7662,  749,  102,    0,
                                     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0],
                                    [101,  697, 1346, 4638, 1962, 8024, 5110, 5117,  679, 6631, 6814,  712,
                                     7608, 4638,  122,  120,  124,  102,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                                     0,    0,    0],
                                    [101, 5131, 2228, 4567, 6639, 3221, 4507,  754, 5131, 2228, 4567, 4638,
                                     4868, 5307, 4567, 1359, 8024, 2471, 6629, 6639, 6956, 3971, 4550,  511,
                                     2697, 3381, 8024, 2523,  679, 2159, 3211, 2689, 1394, 8024, 6117, 3890,
                                     2542, 4384,  679, 1962, 8024, 3297, 1400, 1776, 4564, 3766, 3300, 2971,
                                     1169, 8024,  102]],)
    contex_mask_tensor = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    contex_seq_tensor = torch.tensor([[101,   101,   872,  6821,  3416,  6432,  2769,  2552,  2658,  1962,
                                       1914,   749,   102,  3300,  1377,  5543,  8024,  5165,  2476,   738,
                                       1377,  6117,  5131,  7770,   102,  2208,  1391,  1914,  7623,  1962,
                                       6820,  3221,   671,  1921,   676,  7623,  1962,  8024,  2769,  2218,
                                       3221,  5165,  2476,  7350,  8024,  1045,  2586,  8024,  6117,  5131,
                                       7770,   102,     0,     0,     0,     0,     0,     0,     0,     0],
                                     [101,   101,  2476,  1920,  1923,  3219,  1921,  2769,  2563,  7309,
                                      872,   749,  2769,  5455,  1449,  4638,  1326,  2154,   671,  4684,
                                      1510,   702,   679,   977,   102,     0,     0,     0,     0,     0,
                                      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                      0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                      0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                                      [101,   101,  1920,  1923,   511,  2769,  3844,  6117,  5273,  6117,
                                       5131,   124,   119,  8398,  8024,  3221,  1415,  6206,  1121,  5790,
                                       7030,   102,  3221,  5131,  1265,  1416,   102,  3221,  4638,  8024,
                                       1377,  2769,  4638,  6117,  5131,   811,  3844,  4638,  6963,  1762,
                                       127, 11202,  8167,   102,     0,     0,     0,     0,     0,     0,
                                       0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                                      [101,   101,  2769,  2682,  7309,   678,  1600,  6486,  3841,   833,
                                       1285,  7770,  6117,  5131,  1408,   102,   738,  1419,  3300,  4178,
                                       7030,  4638,   511,   679,  6814,  3221,   809,  3490,  4289,  6028,
                                       4635,   711,   712,   102,   671,  5663,  6117,  5131,  2582,   720,
                                       4664,  3844,  3683,  6772,  1394,  4415,  1450,  8043,  1921,  1921,
                                       2799,  2797,  1922,  4563,   749,   102,     0,     0,     0,     0],
                                      [101,   101,  2476,  1920,  1923,  8024,  2769,  2682,  6206,  2111,
                                       2094,  8024,  6117,  5131,   679,  4937,  2137,  2512,  1510,  1920,
                                       1408,  8043,   102,  6206,  2111,  2094,   722,  1184,  1044,  2802,
                                       5536,  2270,  5162,  8024,  6444,  1962,  6117,  5131,  8024,  1086,
                                       6206,   102,     0,     0,     0,     0,     0,     0,     0,     0,
                                       0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                                      [101,   101,  3300,  1377,  5543,  8024,  5165,  2476,   738,  1377,
                                       6117,  5131,  7770,   102,  2208,  1391,  1914,  7623,  1962,  6820,
                                       3221,   671,  1921,   676,  7623,  1962,  8024,  2769,  2218,  3221,
                                       5165,  2476,  7350,  8024,  1045,  2586,  8024,  6117,  5131,  7770,
                                       102,  2456,  6379,  3221,  2208,  7030,  1914,  7623,  8024,  1377,
                                       809,   678,  1286,  1217,   671,  7623,   102,     0,     0,     0],
                                      [101,   101,  1059,  7931,  5106,  1469,  3249,  6858,  7481,  5106,
                                       3300,  1277,  1166,  1408,  8043,  7672,  1928,  3221,   697,  1346,
                                       7481,  1962,  8024,  6820,  3221,  1059,  7931,  5106,  4638,  1962,
                                       511,   102,     0,     0,     0,     0,     0,     0,     0,     0,
                                       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                                       0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                                      [101,   101,  3221,   679,  3221,  1100,  1921,  3766,  3800,  2692,
                                       924,  3265,  1355,  1117,  1450,  8043,  8024,  2772,  6117,  3890,
                                       897,  2418,   679,  1168,  6639,  6956,  8043,   102,  3300,  1377,
                                       5543,  8024,  6206,   924,  3265,  8024,  3315,  6716,  5131,  2228,
                                       4567,   782,  2697,  6230,  6826,  7162,   102,  5131,  2228,  4567,
                                       6639,  3221,   784,   720,  3416,  4638,  4568,  4307,  8043,   102]])
    intent_tensor = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0.],
                                  [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0., 0., 0., 0.]])

    # forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,intent_tensor=None, context_seq_tensor=None, context_mask_tensor=None):
    model(word_seq_tensor, word_mask_tensor, tag_seq_tensor,
          tag_mask_tensor, intent_tensor, contex_seq_tensor, contex_mask_tensor)