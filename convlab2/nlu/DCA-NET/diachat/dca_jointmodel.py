# coding:utf-8
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from crf import CRF
from dynamic_rnn import DynamicLSTM
import time
# from data_util import config
from transformers import BertModel
import model_config as config
# from model_config import HIDENSIZE

# device = torch.device("cuda:1" if torch.cuda.is_available() and use_gpu else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() and use_gpu else "cpu")
class Joint_model(nn.Module):
    # def __init__(self, _, hidden_dim, batch_size, max_length, n_class, n_tag, embedding_matrix):
    def __init__(self, model_config,device, n_tag, n_class,max_length,max_context_len, intent_weight=None):
        super(Joint_model, self).__init__()
        self.config=model_config
        self.batch_size = model_config['batch_size']
        self.device=device
        self.max_length = 60+2
        self.n_class = n_class
        self.n_tag = n_tag
        self.bert = BertModel.from_pretrained( model_config['pretrained_weights'])
        self.intent_weight =intent_weight if intent_weight is not None else torch.tensor([1.] * n_class)

        self.hidden_dim = config.HIDENSIZE
        self.LayerNorm = LayerNorm(self.hidden_dim,eps=1e-12)
        
        self.LayerNorm2 = LayerNorm(self.hidden_dim,eps=1e-12)
        self.LayerNorm3 = LayerNorm(self.hidden_dim,eps=1e-12)
        # layer_norm = nn.LayerNorm(embedding_dim)
        self.emb_dim=self.bert.config.hidden_size
        self.emb_drop = nn.Dropout(0.8)
        # self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        # self.embed.weight.requires_grad = True
        self.biLSTM = DynamicLSTM(self.emb_dim, self.hidden_dim // 2, bidirectional=True, batch_first=True,
                                  dropout=config.LSTM_DROPOUT, num_layers=1)
        
        self.intent_fc = nn.Linear(self.hidden_dim, self.n_class)
        self.slot_fc = nn.Linear(self.hidden_dim, self.n_tag)

        # if self.config["context"]:
        #     self.intent_fc = nn.Linear(self.hidden_dim*2, self.n_class)
        #     self.slot_fc = nn.Linear(self.hidden_dim*2, self.n_tag)
            

        self.I_S_Emb = Label_Attention(self.intent_fc, self.slot_fc)

        self.T_block1 = I_S_Block(self.intent_fc, self.slot_fc, self.hidden_dim)
        self.T_block2 = I_S_Block(self.intent_fc, self.slot_fc, self.hidden_dim)
        self.T_block3 = I_S_Block(self.intent_fc, self.slot_fc, self.hidden_dim)
        self.crflayer = CRF(self.n_tag)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)

    def forward_logit(self, x, mask,total_length=62):
        # x, x_char = x
        x_len = torch.sum(x != 0, dim=-1)
        # x_emb = self.emb_drop(self.embed(x))
        x_emb=self.bert(input_ids=x,attention_mask=mask)[0]

        H, (_, _) = self.biLSTM(x_emb, x_len, total_length=total_length)
                
        H_I, H_S = self.I_S_Emb(H, H, mask)
        H_I, H_S = self.T_block1(H_I + H, H_S + H, mask)
        H_I_1, H_S_1 = self.I_S_Emb(H_I, H_S, mask)
        H_I, H_S = self.T_block2(H_I + H_I_1, H_S + H_S_1, mask)

        intent_input = F.max_pool1d((H_I + H).transpose(1, 2), H_I.size(1)).squeeze(2)
        return intent_input, H_S + H

    def _cal_logit(self,intent_input,slot_input):
        logits_intent = self.intent_fc(intent_input)
        logits_slot = self.slot_fc(slot_input)

        return logits_intent, logits_slot

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask,slot_mask):
        mask = mask[:, 0:logits_slot.size(1)]
        slot_label = slot_label[:, 0:logits_slot.size(1)]        #[0, 0, 0, 0, 8, 9, 9, 9, 0, 0, 0, 10, 11, 8, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        logits_slot = logits_slot.transpose(1, 0)  # [8, 62, 473] -> [62, 8, 473]
        slot_label = slot_label.transpose(1, 0)  # [8, 62] -> [62,8]
        mask = mask.transpose(1, 0)   ## [8, 62] -> [62,8]       #[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,0, 0,
        slot_mask=slot_mask.transpose(1,0)  #[8, 62] -> [62,8]   #[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 
        loss_intent = self.criterion(logits_intent, intent_label)
        # loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0] 
        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]/ logits_intent.size()[1]

        return loss_intent, loss_slot

    def forward(self,x, word_mask, slot_label, slot_mask,intent_label,context_seq,context_mask):
         # word_mask 1是token位置  0是padding位置  cls算1
        
        intent_input,slot_input=self.forward_logit(x, mask=word_mask , total_length=x.size(1))

        if context_seq is not None and context_mask is not None:
            # print("context_seq is not None and context_mask is not None")
            # print("context_seq",context_seq.size())
            # print("context_mask",context_mask.size())
            

            c_intent_input, c_slot_input = self.forward_logit(context_seq, mask=context_mask,total_length=context_seq.size(1))
            # print("c_intent_input",c_intent_input.size())
            
            # intent_input = torch.cat((c_intent_input,intent_input),dim=-1)
            intent_input = self.LayerNorm2 (c_intent_input)+intent_input
            slot_input = self.LayerNorm3( c_intent_input.unsqueeze(1).repeat(1, slot_input.size(1), 1))+slot_input

        
        logits_intent, logits_slot = self._cal_logit(intent_input,slot_input)

        
        
        
        loss_intent, loss_slot = self.loss1(logits_intent, logits_slot, intent_label, slot_label, mask=word_mask,slot_mask=slot_mask)

        pred_slot = self.pred_intent_slot(logits_slot, mask=word_mask)

        return pred_slot,logits_intent,loss_slot,loss_intent

    # def pred_intent_slot(self, logits_slot, slot_mask):
    #     # dev or test
    #     slot_mask = slot_mask[:, 0:logits_slot.size(1)]
    #     slot_mask = slot_mask.transpose(1, 0)
    #     logits_slot = logits_slot.transpose(1, 0)
    #     # pred_intent = torch.max(logits_intent, 1)[1]
    #     pred_slot = self.crflayer.decode(logits_slot, mask=slot_mask)
    #     for i in range(len(pred_slot)):
    #         pred_slot[i] = [x.tolist() for x in pred_slot[i]]
    #     return pred_slot
    def pred_intent_slot(self, logits_slot, mask):
        # dev or test
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        # pred_intent = torch.max(logits_intent, 1)[1]
        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        for i in range(len(pred_slot)):
            pred_slot[i] = [x.tolist() for x in pred_slot[i]]
        return pred_slot


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)).to(config.device)
        self.bias = nn.Parameter(torch.zeros(hidden_size)).to(config.device)
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super(Intermediate, self).__init__()
        self.dense_in = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states_in):
        hidden_states = self.dense_in(hidden_states_in)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + hidden_states_in)
        return hidden_states


class Intermediate_I_S(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super(Intermediate_I_S, self).__init__()
        self.dense_in = nn.Linear(hidden_size * 6, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm_I = LayerNorm(hidden_size, eps=1e-12)
        self.LayerNorm_S = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states_I, hidden_states_S):
        hidden_states_in = torch.cat([hidden_states_I, hidden_states_S], dim=2)
        batch_size, max_length, hidden_size = hidden_states_in.size()
        h_pad = torch.zeros(batch_size, 1, hidden_size).to(config.device)
        # if use_gpu and torch.cuda.is_available():
        #     h_pad = h_pad.cuda()
        h_left = torch.cat([h_pad, hidden_states_in[:, :max_length - 1, :]], dim=1)
        h_right = torch.cat([hidden_states_in[:, 1:, :], h_pad], dim=1)
        hidden_states_in = torch.cat([hidden_states_in, h_left, h_right], dim=2)

        hidden_states = self.dense_in(hidden_states_in)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states_I_NEW = self.LayerNorm_I(hidden_states + hidden_states_I)
        hidden_states_S_NEW = self.LayerNorm_S(hidden_states + hidden_states_S)
        return hidden_states_I_NEW, hidden_states_S_NEW


class I_S_Block(nn.Module):
    def __init__(self, intent_emb, slot_emb, hidden_size):
        super(I_S_Block, self).__init__()
        self.I_S_Attention = I_S_SelfAttention(hidden_size, 2 * hidden_size, hidden_size)
        self.I_Out = SelfOutput(hidden_size, config.attention_dropout)
        self.S_Out = SelfOutput(hidden_size, config.attention_dropout)
        self.I_S_Feed_forward = Intermediate_I_S(hidden_size, hidden_size)

    def forward(self, H_intent_input, H_slot_input, mask):
        H_slot, H_intent = self.I_S_Attention(H_intent_input, H_slot_input, mask)
        H_slot = self.S_Out(H_slot, H_slot_input)
        H_intent = self.I_Out(H_intent, H_intent_input)
        H_intent, H_slot = self.I_S_Feed_forward(H_intent, H_slot)

        return H_intent, H_slot


class Label_Attention(nn.Module):
    def __init__(self, intent_emb, slot_emb):
        super(Label_Attention, self).__init__()

        self.W_intent_emb = intent_emb.weight
        self.W_slot_emb = slot_emb.weight

    def forward(self, input_intent, input_slot, mask):
        # print("input_intent size=", input_intent.size())
        # print("input_slot size=", input_slot.size())
        # print("W_intent_emb size=", self.W_intent_emb.size())
        intent_score = torch.matmul(input_intent, self.W_intent_emb.t())
        slot_score = torch.matmul(input_slot, self.W_slot_emb.t())
        intent_probs = nn.Softmax(dim=-1)(intent_score)
        slot_probs = nn.Softmax(dim=-1)(slot_score)
        intent_res = torch.matmul(intent_probs, self.W_intent_emb)
        slot_res = torch.matmul(slot_probs, self.W_slot_emb)

        return intent_res, slot_res


class I_S_SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(I_S_SelfAttention, self).__init__()

        self.num_attention_heads = 8
        self.attention_head_size = int(hidden_size / self.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.out_size = out_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.query_slot = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.key_slot = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.out_size)
        self.value_slot = nn.Linear(input_size, self.out_size)
        self.dropout = nn.Dropout(config.attention_dropout)

    def transpose_for_scores(self, x):
        last_dim = int(x.size()[-1] / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, last_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, intent, slot, mask):
        # mask 1是token位置  0是padding位置
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - extended_attention_mask) * -10000.0

        mixed_query_layer = self.query(intent)
        mixed_key_layer = self.key(slot)
        mixed_value_layer = self.value(slot)

        mixed_query_layer_slot = self.query_slot(slot)
        mixed_key_layer_slot = self.key_slot(intent)
        mixed_value_layer_slot = self.value_slot(intent)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer_slot = self.transpose_for_scores(mixed_query_layer_slot)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        key_layer_slot = self.transpose_for_scores(mixed_key_layer_slot)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        value_layer_slot = self.transpose_for_scores(mixed_value_layer_slot)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores_slot = torch.matmul(query_slot, key_slot.transpose(1,0))
        attention_scores_slot = torch.matmul(query_layer_slot, key_layer_slot.transpose(-1, -2))
        attention_scores_slot = attention_scores_slot / math.sqrt(self.attention_head_size)
        attention_scores_intent = attention_scores + attention_mask  # 报错

        attention_scores_slot = attention_scores_slot + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs_slot = nn.Softmax(dim=-1)(attention_scores_slot)
        attention_probs_intent = nn.Softmax(dim=-1)(attention_scores_intent)

        attention_probs_slot = self.dropout(attention_probs_slot)
        attention_probs_intent = self.dropout(attention_probs_intent)

        context_layer_slot = torch.matmul(attention_probs_slot, value_layer_slot)
        context_layer_intent = torch.matmul(attention_probs_intent, value_layer)

        context_layer = context_layer_slot.permute(0, 2, 1, 3).contiguous()
        context_layer_intent = context_layer_intent.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.out_size,)
        new_context_layer_shape_intent = context_layer_intent.size()[:-2] + (self.out_size,)

        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer_intent = context_layer_intent.view(*new_context_layer_shape_intent)
        return context_layer, context_layer_intent
if __name__ == "__main__":
    # print(config.emb_dorpout)
    import json
    import os
    # print(os.path.abspath(__file__))
    # print(os.getcwd())
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(cur_dir, 'config/all_context.json')
    # print(config_file)
    conf = json.load(open(config_file))
    model_conf = conf["model"]
    # print(model_conf)
    device="cuda:1" 
    model = Joint_model(model_conf, device, 470, 58 ,49,60)
    model.to("cuda:1" )
    # summary(model(),(8,51))

    # TODO crf这个包 不能以0打头
    tag_mask_tensor = torch.tensor([[
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0
    ],
                                    [
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                    ],
                                    [
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                    ],
                                    [
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                    ],
                                    [
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                    ],
                                    [
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                    ],
                                    [
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                    ],
                                    [
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0
                                    ]])
    tag_seq_tensor = torch.tensor(
        [[
            0, 0, 0, 0, 8, 9, 9, 9, 0, 0, 0, 10, 11, 8, 9, 9, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0
        ],
         [
             0, 0, 0, 150, 151, 151, 0, 0, 150, 151, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 75, 75, 75, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 144, 145, 0, 144, 145, 145, 145,
             145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0
         ],
         [
             0, 304, 305, 0, 0, 0, 0, 0, 0, 0, 158, 159, 159, 159, 0, 0, 84, 0,
             16, 17, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             0, 78, 79, 0, 0, 0, 163, 208, 264, 298, 298, 298, 298, 298, 298,
             298, 298, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0
         ]])
    word_mask_tensor = torch.tensor(
        [[
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0
        ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0
         ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0
         ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0
         ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0
         ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0
         ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0
         ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1
         ]])
    word_seq_tensor = torch.tensor(
        [[
            101, 2456, 6379, 3221, 2208, 7030, 1914, 7623, 8024, 1377, 809,
            678, 1286, 1217, 671, 7623, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ],
         [
             101, 2644, 3300, 6117, 1327, 7770, 2772, 5442, 6577, 6117, 1408,
             8043, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             101, 1963, 3362, 2971, 1169, 6821, 3416, 1962, 117, 1377, 809,
             2714, 2714, 1121, 7030, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0
         ],
         [
             101, 1008, 872, 6821, 3416, 3683, 6772, 4937, 2137, 4638, 8024,
             4958, 5592, 1469, 3241, 7623, 123, 2207, 3198, 2218, 1377, 809,
             4638, 511, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             101, 2769, 6230, 2533, 2802, 5536, 2270, 5162, 738, 679, 671,
             2137, 4638, 5543, 4937, 2137, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0
         ],
         [
             101, 1217, 7623, 1391, 1567, 3683, 6772, 1962, 8024, 2769, 1282,
             671, 4157, 1288, 1391, 4638, 7649, 8024, 4385, 1762, 7662, 749,
             102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             101, 697, 1346, 4638, 1962, 8024, 5110, 5117, 679, 6631, 6814,
             712, 7608, 4638, 122, 120, 124, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0
         ],
         [
             101, 5131, 2228, 4567, 6639, 3221, 4507, 754, 5131, 2228, 4567,
             4638, 4868, 5307, 4567, 1359, 8024, 2471, 6629, 6639, 6956, 3971,
             4550, 511, 2697, 3381, 8024, 2523, 679, 2159, 3211, 2689, 1394,
             8024, 6117, 3890, 2542, 4384, 679, 1962, 8024, 3297, 1400, 1776,
             4564, 3766, 3300, 2971, 1169, 8024, 102
         ]], )
    context_mask_tensor = torch.tensor(
        [[
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0
        ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0
         ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0
         ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
         ]])
    context_seq_tensor = torch.tensor(
        [[
            101, 101, 872, 6821, 3416, 6432, 2769, 2552, 2658, 1962, 1914, 749,
            102, 3300, 1377, 5543, 8024, 5165, 2476, 738, 1377, 6117, 5131,
            7770, 102, 2208, 1391, 1914, 7623, 1962, 6820, 3221, 671, 1921,
            676, 7623, 1962, 8024, 2769, 2218, 3221, 5165, 2476, 7350, 8024,
            1045, 2586, 8024, 6117, 5131, 7770, 102, 0, 0, 0, 0, 0, 0, 0, 0
        ],
         [
             101, 101, 2476, 1920, 1923, 3219, 1921, 2769, 2563, 7309, 872,
             749, 2769, 5455, 1449, 4638, 1326, 2154, 671, 4684, 1510, 702,
             679, 977, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             101, 101, 1920, 1923, 511, 2769, 3844, 6117, 5273, 6117, 5131,
             124, 119, 8398, 8024, 3221, 1415, 6206, 1121, 5790, 7030, 102,
             3221, 5131, 1265, 1416, 102, 3221, 4638, 8024, 1377, 2769, 4638,
             6117, 5131, 811, 3844, 4638, 6963, 1762, 127, 11202, 8167, 102, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             101, 101, 2769, 2682, 7309, 678, 1600, 6486, 3841, 833, 1285,
             7770, 6117, 5131, 1408, 102, 738, 1419, 3300, 4178, 7030, 4638,
             511, 679, 6814, 3221, 809, 3490, 4289, 6028, 4635, 711, 712, 102,
             671, 5663, 6117, 5131, 2582, 720, 4664, 3844, 3683, 6772, 1394,
             4415, 1450, 8043, 1921, 1921, 2799, 2797, 1922, 4563, 749, 102, 0,
             0, 0, 0
         ],
         [
             101, 101, 2476, 1920, 1923, 8024, 2769, 2682, 6206, 2111, 2094,
             8024, 6117, 5131, 679, 4937, 2137, 2512, 1510, 1920, 1408, 8043,
             102, 6206, 2111, 2094, 722, 1184, 1044, 2802, 5536, 2270, 5162,
             8024, 6444, 1962, 6117, 5131, 8024, 1086, 6206, 102, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
         ],
         [
             101, 101, 3300, 1377, 5543, 8024, 5165, 2476, 738, 1377, 6117,
             5131, 7770, 102, 2208, 1391, 1914, 7623, 1962, 6820, 3221, 671,
             1921, 676, 7623, 1962, 8024, 2769, 2218, 3221, 5165, 2476, 7350,
             8024, 1045, 2586, 8024, 6117, 5131, 7770, 102, 2456, 6379, 3221,
             2208, 7030, 1914, 7623, 8024, 1377, 809, 678, 1286, 1217, 671,
             7623, 102, 0, 0, 0
         ],
         [
             101, 101, 1059, 7931, 5106, 1469, 3249, 6858, 7481, 5106, 3300,
             1277, 1166, 1408, 8043, 7672, 1928, 3221, 697, 1346, 7481, 1962,
             8024, 6820, 3221, 1059, 7931, 5106, 4638, 1962, 511, 102, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0
         ],
         [
             101, 101, 3221, 679, 3221, 1100, 1921, 3766, 3800, 2692, 924,
             3265, 1355, 1117, 1450, 8043, 8024, 2772, 6117, 3890, 897, 2418,
             679, 1168, 6639, 6956, 8043, 102, 3300, 1377, 5543, 8024, 6206,
             924, 3265, 8024, 3315, 6716, 5131, 2228, 4567, 782, 2697, 6230,
             6826, 7162, 102, 5131, 2228, 4567, 6639, 3221, 784, 720, 3416,
             4638, 4568, 4307, 8043, 102
         ]])
    intent_tensor = torch.tensor([[
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.
    ],
                                  [
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0.
                                  ],
                                  [
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0.
                                  ],
                                  [
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0.
                                  ],
                                  [
                                      0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0.
                                  ],
                                  [
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0.
                                  ],
                                  [
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0.
                                  ],
                                  [
                                      0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 0., 0.
                                  ]])

    # forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,intent_tensor=None, context_seq_tensor=None, context_mask_tensor=None):
    model(word_seq_tensor.to(device), word_mask_tensor.to(device), tag_seq_tensor.to(device), tag_mask_tensor.to(device),
          intent_tensor.to(device), context_seq_tensor.to(device), context_mask_tensor.to(device))
    # summary(model,[(51,),(51,),(51,),(51,) ,(58,),(58,),(58,)],dtypes=[torch.int,torch.bool,torch.int,torch.bool,torch.int,torch.int,torch.bool], device = "cuda:3")
    print("done")