# slot 解码时加入注意力机制
# slot 解码时加入的注意力机制增加mask
# intetn 添加muse注意力 并mask
# intetn 添加muse注意力 并mask 加入残差链接
# slot 添加残差链接

import torch
from torch import nn
from torch.autograd import Variable
from transformers import BertModel
from torchcrf import CRF
from model_config import *
import copy
import torch.nn.functional as F
# from fastNLP.modules.torch.attention import MultiHeadAttention
# from fastNLP.models.torch.sequence_labeling import *
import math

import numpy as np

from torch.nn import init

class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)



        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        其中，Q为Query、K为Key、V为Value。Q、K、V是从哪儿来的呢？Q、K、V其实都是从同样的输入矩阵X线性变换而来的
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out
class Depth_Pointwise_Conv1d(nn.Module):
    def __init__(self,in_ch,out_ch,k):
        super().__init__()
        if(k==1):
            self.depth_conv=nn.Identity()
        else:
            self.depth_conv=nn.Conv1d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding=k//2
                )
        self.pointwise_conv=nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1
        )
    def forward(self,x):
        out=self.pointwise_conv(self.depth_conv(x))
        return out

class MUSEAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):


        super(MUSEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.conv1=Depth_Pointwise_Conv1d(h * d_v, d_model,1)
        self.conv3=Depth_Pointwise_Conv1d(h * d_v, d_model,3)
        self.conv5=Depth_Pointwise_Conv1d(h * d_v, d_model,5)
        self.dy_paras=nn.Parameter(torch.ones(3))
        self.softmax=nn.Softmax(-1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        #Self Attention
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)   # masked_fill 用 value填充 mask为1的地方 
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        v2=v.permute(0,1,3,2).contiguous().view(b_s,-1,nk) #bs,dim,n
        self.dy_paras=nn.Parameter(self.softmax(self.dy_paras))
        out2=self.dy_paras[0]*self.conv1(v2)+self.dy_paras[1]*self.conv3(v2)+self.dy_paras[2]*self.conv5(v2)
        out2=out2.permute(0,2,1) #bs.n.dim

        out=out+out2
        return out
    

# 定义一个clones函数，来更方便的将某个结构复制若干份
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    """ 
    实现子层连接结构的类

    """
    def __init__(self, size, dropout=0.1):
        """
            size = sublayer_out的维度
            dropout = 0.1
        """
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        # 原paper的方案
        #sublayer_out = sublayer(x)
        #x_norm = self.norm(x + self.dropout(sublayer_out))

        # 稍加调整的版本
        sublayer_out = sublayer(x)
        sublayer_out = self.dropout(sublayer_out)
        x_norm = x + self.norm(sublayer_out)
        return x_norm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        位置编码器类的初始化函数
        
        共有三个参数，分别是
        d_model：词嵌入维度
        dropout: dropout触发比率
        max_len：每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings
        # 注意下面代码的计算方式与公式中给出的是不同的，但是是等价的，你可以尝试简单推导证明一下。
        # 这样计算是为了避免中间的数值计算结果超出float的范围，
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class AttentionLayer(nn.Module):

    def __init__(selfu, input_size, key_dim, value_dim, bias=False):
        """
        可用于LSTM2LSTM的序列到序列模型的decode过程中，该attention是在decode过程中根据上一个step的hidden计算对encoder结果的attention

        :param int input_size: 输入的大小
        :param int key_dim: 一般就是encoder_output输出的维度
        :param int value_dim: 输出的大小维度, 一般就是decoder hidden的大小
        :param bias:
        """
        super().__init__()

        selfu.input_proj = nn.Linear(input_size, key_dim, bias=bias)
        selfu.output_proj = nn.Linear(input_size + key_dim,
                                      value_dim,
                                      bias=bias)

    def forward(self, input, encode_outputs, encode_mask):
        """
        :param input: batch_size x input_size
        :param encode_outputs: batch_size x max_len x key_dim
        :param encode_mask: batch_size x max_len, 为0的地方为padding
        :return: hidden: batch_size x value_dim, scores: batch_size x max_len, normalized过的
        """

        # x: bsz x encode_hidden_size
        x = self.input_proj(input)

        # compute attention
        attn_scores = torch.matmul(encode_outputs,
                                   x.unsqueeze(-1)).squeeze(-1)  # b x max_len

        # don't attend over padding
        if encode_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encode_mask.eq(0), float('-inf')).type_as(
                    attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=-1)  # srclen x bsz

        # sum weighted sources
        x = torch.matmul(attn_scores.unsqueeze(1),
                         encode_outputs).squeeze(1)  # b x encode_hidden_size

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class Slot_Enc(nn.Module):

    def __init__(self, in_size, lstm_hidden_size) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size * 2)

        if 2 * lstm_hidden_size != in_size:
            # self.use_1x1conv = True
            self.conv1x1 = nn.Conv1d(in_size,
                                     lstm_hidden_size * 2,
                                     kernel_size=1,
                                     stride=1)
        else:
            self.conv1x1 = None
        self.dropouts = clones(nn.Dropout(DROPOUT),2)

    def forward(self, x):
        """
            x_in : [batch,seqlen,emb_size]
        """
        x = self.dropouts[0](x)
        y, _ = self.lstm(x)  # batch, seqlen, hiddensize*2
        y = self.dropouts[0](y)
        if self.conv1x1:
            x = self.conv1x1(x.transpose(-1,-2)).transpose(-1,-2)  # 1D卷积 时通道在  (batch , input_size , seq_len) input_size位置为通道  与RNN不一样
        #add & norm
        y += x
        y = self.layer_norm(y)
        return y  # batch, seqlen, hiddensize*2


class Slot_Dec(nn.Module):

   

    def __init__(self, lstm_hidden_size, device) -> None:
        super().__init__()
        self.hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(input_size=lstm_hidden_size * 5,
                            hidden_size=lstm_hidden_size,
                            num_layers=1)
        self.device = device
        self.dropouts = clones(nn.Dropout(DROPOUT),2)
        # self.fc = nn.Linear(lstm_hidden_size, slot_dim)  # need ?
        self.atte = AttentionLayer(input_size=lstm_hidden_size * 5,key_dim=lstm_hidden_size * 4,value_dim=lstm_hidden_size)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)
    def forward(self, x, hi):
        """
        x : []
        return [batch , seqlen , lstm_hidden_size]
        """
        batch = x.size(0)
        seqlenth = x.size(1)
        dec_init_out = torch.zeros(batch, 1, self.hidden_size).to(self.device)
        hidden_state = (torch.zeros(1, 1, self.hidden_size).to(self.device),
                        torch.zeros(1, 1, self.hidden_size).to(self.device))
        x = torch.cat((x, hi), dim=-1)
        x = x.transpose(1, 0)
        x = self.dropouts[0](x)
        all_out = []
        for i in range(seqlenth):
            if i == 0:
                out, hidden_state = self.lstm(torch.cat((x[i].unsqueeze(1), dec_init_out), dim=-1),
                                                                                    hidden_state)
            else:  #cat完 batch,1,1000
                out, hidden_state = self.lstm(
                    torch.cat((x[i].unsqueeze(1), out), dim=-1), hidden_state)
            all_out.append(out)
        output = torch.cat(all_out, dim=1)
        output = self.dropouts[0](output)
        # res = self.fc(output)
        # return res # 16,50，130
        return output  # [batch , seqlen , lstm_hidden_size]


class Intent_Enc(nn.Module):

    def __init__(self, in_size, lstm_hidden_size) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True,
                            dropout=DROPOUT          
                            )
        self.layer_norm = nn.LayerNorm(lstm_hidden_size * 2)

        if 2 * lstm_hidden_size != in_size:
            # self.use_1x1conv = True
            self.conv1x1 = nn.Conv1d(in_size,
                                     lstm_hidden_size * 2,
                                     kernel_size = 1,
                                     stride = 1)
        else:
            self.conv1x1 = None
        self.dropouts = clones(nn.Dropout(DROPOUT),2)

    def forward(self, x):
        """
            x : [batch,seqlen,emb_size]
        """
        x = self.dropouts[0](x)
        y, _ = self.lstm(x)  # batch, seqlen, hiddensize*2
        y = self.dropouts[1](y)
        # print(y.shape)
        # print(x.shape)
        if self.conv1x1:
             x = self.conv1x1(x.transpose(-1,-2)).transpose(-1,-2)  # 1D卷积 时通道在  (batch , input_size , seq_len) input_size位置为通道  与RNN不一样
            # print(x.shape)
        #add & norm
        y += x
        y = self.layer_norm(y)
        return y  # batch, seqlen, hiddensize*2

class Intent_Dec(nn.Module):

    def __init__(self, lstm_hidden_size, device) -> None:
        """
        lstm_hidden_size : encoder out_dim
        device 
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=lstm_hidden_size * 4, 
                            hidden_size=lstm_hidden_size,
                            batch_first=True,
                            num_layers=1)
        self.muse_attention=MUSEAttention(lstm_hidden_size,lstm_hidden_size,lstm_hidden_size,MUSE_HEAD)  # 8头注意力
        self.device = device
        self.dropouts = clones(nn.Dropout(DROPOUT),2)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)
        self.layer1= nn.LayerNorm(lstm_hidden_size,lstm_hidden_size)
        self.layer2 = nn.LayerNorm(lstm_hidden_size,lstm_hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        # self.a_pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x, hs, real_len , mask):
        """
        x:
        hs:
        return： state real长度最后一时刻输出 (batch , lstm_hidden_size)
        """
        batch = x.size(0)
        # real_len = torch.tensor(real_len).to(self.device)
        x = torch.cat((x, hs), dim=-1)
        x = self.dropouts[0](x)
        x, _ = self.lstm(x)      
        x = self.dropouts[1](x)  # batch,len,lstm_hidden_size
        #自注意力
        
        index = torch.arange(batch).long().to(self.device)  # 0-15 
        # 表示第一个维度取index的 第二个维度取real_len-1的，第三个取满足前两个条件的所有数， 
        # 表示real长度的最后一时刻输出   
        # mask.unsqueeze(-1).unsqueeze(1).repeat(1,2,1,3).permute(0,1,3,2)
        if mask is not None:
            y=self.muse_attention(x,x,x,mask.eq(0).unsqueeze(-1).unsqueeze(1).repeat(1,MUSE_HEAD,1,x.size(1)).permute(0,1,3,2) )
            # y=self.dropouts[2](y)                                    #   add attention6
            y=self.layer2(y)
            y=self.leakyrelu(self.layer1(y))
            
            x+=y
            x=self.layer_norm(x)
        else:
            y=self.muse_attention(x,x,x)
            # y=self.dropouts[2](y)                                       #   add attention6
            y=self.layer2(y)
            y=self.leakyrelu(self.layer1(y))

            x+=y
            x=self.layer_norm(x)
        
        state = x[index, real_len-1, :]    # (batch , lstm_hidden_size)
        # x=x.permute(0,2,1)
        # x=self.a_pool(x)
        # state=x.squeeze(-1)
        return state 

# class Intent(nn.Module):
#     def __init__(self):
#         super(Intent, self).__init__()
#         self.enc = Intent_Enc(cfg.embedding_size, cfg.lstm_hidden_size).to(device) # 300 ,200
#         self.dec = Intent_Dec(cfg.lstm_hidden_size).to(device)  #200
#         self.share_memory = torch.zeros(cfg.batch, cfg.max_len, cfg.lstm_hidden_size * 2).to(device)

# class Slot(nn.Module):
#     def __init__(self):
#         super(Slot, self).__init__()
#         self.enc = Slot_Enc(cfg.embedding_size, cfg.lstm_hidden_size).to(device)  # embedding_size=300 ,lstm_hidden_size=200
#         self.dec = Slot_Dec(cfg.lstm_hidden_size).to(device)
#         self.share_memory = torch.zeros(cfg.batch, cfg.max_len, cfg.lstm_hidden_size * 2).to(device)  #batch=16  ,max_len=50,lstm_hidden_size=200
class JointBERTCRFLSTM(nn.Module):

    def __init__(self,
                 model_config,
                 device,
                 slot_dim,
                 intent_dim,
                 max_sen_len,
                 max_context_len,
                 intent_weight=None,
                 ):
        super().__init__()
        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.device = device
        self.context = model_config['context']
        self.finetune = model_config['finetune']
        self.context_grad = model_config['context_grad']
        self.hidden_units = model_config['hidden_units']   # hidden_units = 768*2
        self.batch_size = model_config['batch_size']
        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.] * intent_dim)
        self.max_seq_len = max_sen_len + 2
        self.max_context_len = max_context_len # 这个不用加
        print(model_config['pretrained_weights'])
        self.bert = BertModel.from_pretrained( model_config['pretrained_weights'])
        print("bert", self.bert)
        print(self.bert.config)
        self.sublayers_1 = SublayerConnection(size=self.bert.config.hidden_size,dropout=0.1 )  #子层为intent和slot 的 ENCODER 和 decoder
        if USELSTM:
            self.enc_i = Intent_Enc(in_size= self.bert.config.hidden_size , lstm_hidden_size = LSTM_HIDDEN_SIZE ) #  LSTM_HIDDEN_SIZE=512
            self.dec_i = Intent_Dec(LSTM_HIDDEN_SIZE , device)
            self.enc_s = Slot_Enc(self.bert.config.hidden_size , LSTM_HIDDEN_SIZE)
            self.dec_s = Slot_Dec(LSTM_HIDDEN_SIZE, device)
            self.share_memory_i = torch.zeros(self.batch_size , self.max_seq_len , LSTM_HIDDEN_SIZE * 2).to(device)
            self.share_memory_i= torch.zeros(self.batch_size , self.max_seq_len , LSTM_HIDDEN_SIZE * 2).to(device)

        self.dropout = nn.Dropout(model_config['dropout'])

        
        if self.hidden_units > 0:
            if self.context:
                # self.intent_hidden = nn.Linear(2 * (self.bert.config.hidden_size ) , self.hidden_units)
                # self.slot_hidden = nn.Linear(2 * (self.bert.config.hidden_size ),self.hidden_units)
                self.intent_hidden = nn.Linear(2 * (self.bert.config.hidden_size + LSTM_HIDDEN_SIZE) , self.hidden_units)
                self.slot_hidden = nn.Linear(2 * (self.bert.config.hidden_size + LSTM_HIDDEN_SIZE),self.hidden_units)
                self.intent_classifier = nn.Linear(self.hidden_units,self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.hidden_units,self.slot_num_labels)

            else:
                self.intent_hidden = nn.Linear(self.bert.config.hidden_size,self.hidden_units)
                self.slot_hidden = nn.Linear(self.bert.config.hidden_size,self.hidden_units)
                self.intent_classifier = nn.Linear(self.hidden_units,self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.hidden_units,self.slot_num_labels)
            nn.init.xavier_uniform_(self.intent_hidden.weight)
            nn.init.xavier_uniform_(self.slot_hidden.weight)
        else:
            if self.context:
                self.intent_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.slot_num_labels)
            else:
                self.intent_classifier = nn.Linear(self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.bert.config.hidden_size,self.slot_num_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier.weight)

            # self.myattention = AttentionLayer(
            #     input_size=self.bert.config.hidden_size,
            #     key_dim=self.bert.config.hidden_size,
            #     value_dim=self.bert.config.hidden_size)
        # self.myMultiattention = MultiHeadAttention(self.bert.config.hidden_size)
        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)

        if LAST_ADD_CRF:
            print("----BERT+CRF------")
            print("----请注意核对后处理.py 以及 train.py的model.forward()使用-------")
            self.crf = CRF(self.slot_num_labels, batch_first=True)
            # self.slot_loss_fct = self.crf.forward
        else:
            print("请使用CRF")
            exit()

    def forward(
        self,
        word_seq_tensor,
        word_mask_tensor,
        tag_seq_tensor=None,
        tag_mask_tensor=None,  # 他们四个都是一样的，比如torch.Size([8, 51])
        intent_tensor=None,
        context_seq_tensor=None,
        context_mask_tensor=None
    ):  # torch.Size([8, 58])  torch.Size([8, 60])  torch.Size([8, 60])
        '''
        return tag_seq_id, intent_logits, (crf_slot_loss), (intent_loss),
        '''
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(input_ids=word_seq_tensor,
                                    attention_mask=word_mask_tensor)
        else: 
            #微调bert
            outputs = self.bert(input_ids=word_seq_tensor,
                                attention_mask=word_mask_tensor)

        bert_sequence_output = outputs[0]  # [8 ，51， 768]  [batch_size , seq_len, bert_hiddensize]
        bert_pooled_output = outputs[1]  # [8,768]    [batch_size, bert_hiddensize]

        hs = self.enc_s(bert_sequence_output)
        self.share_memory_s = hs.clone()
        hi = self.enc_i(bert_sequence_output)
        self.share_memory_i = hi.clone()
        bimodel_slot_output = self.dec_s(hs , self.share_memory_i.detach())   # [batch, seqlen, lstm_hidden_size]
        # if ATTENTION_MASK:
        #     bimodel_slot_output = self.dec_s(hs , self.share_memory_i.detach(),word_mask_tensor)   # [batch, seqlen, lstm_hidden_size]
        # else:
        #     bimodel_slot_output = self.dec_s(hs , self.share_memory_i.detach(),None)   # [batch, seqlen, lstm_hidden_size]
        # print(bimodel_slot_output.size())
        active_len_seq = word_mask_tensor.sum(dim=-1)  # 尝试：含cls 和 seq TODO 另一种做法不含
        # print(active_len_seq)
        if MUSE_MASK:
            bimodel_intent_output = self.dec_i(hi , self.share_memory_s.detach() ,active_len_seq ,word_mask_tensor)   # [batch , lstm_hidden_size]
        else:
            bimodel_intent_output = self.dec_i(hi , self.share_memory_s.detach() ,active_len_seq ,None)   # [batch , lstm_hidden_size]
        # print(bimodel_intent_output.shape)
        '''
        拼接法
        '''
        sequence_output = torch.cat((bert_sequence_output,bimodel_slot_output),dim=-1) # 8 51 1280
        pooled_output = torch.cat((bert_pooled_output,bimodel_intent_output),dim=-1) # 8 1280
        pass


        if self.context and (context_seq_tensor is not None):
            if not self.finetune or not self.context_grad:
                # with torch.no_grad():
                #     context_output = self.bert(
                #         input_ids=context_seq_tensor,
                #         attention_mask=context_mask_tensor)[1]
                pass
            else:
                outputs=self.bert(input_ids=context_seq_tensor,attention_mask=context_mask_tensor)
                bert_context_seqout , bert_context_poolout =outputs[0],outputs[1]   # 取【1位置】[8,768]
                chs = self.enc_s(bert_context_seqout)
                self.share_memory_cs = chs.clone()
                chi = self.enc_i(bert_context_seqout)
                # self.share_memory_ci = chi.clone()
                # bcso = self.dec_s( chs ,self.share_memory_ci.detach())
                active_len_context = context_mask_tensor.sum(dim=-1)
                if MUSE_MASK:
                    bcio = self.dec_i(chi , self.share_memory_cs.detach() ,active_len_context ,context_mask_tensor)
                else:
                    bcio = self.dec_i(chi , self.share_memory_cs.detach() ,active_len_context ,None)
                context_output = torch.cat((bert_context_poolout,bcio),dim=-1)
            sequence_output = torch.cat(
                # context_output.unsqueeze(1) torch.Size([8, 1, 768]) # .repeat(通道的重复倍数1, 行的重复倍数sequence_output.size(1), 列的重复倍数1) 后torch.Size([8, 51, 768])
                [
                    context_output.unsqueeze(1).repeat(
                        1, sequence_output.size(1), 1),  # unsqueeze()函数起升维的作用
                    sequence_output
                ],
                dim=-1)  # torch.Size([8, 51, 1536])
            pooled_output = torch.cat([context_output, pooled_output],
                                      dim=-1)  # [8,1536]
            # print(sequence_output.shape)
            # print(pooled_output.size())
        if self.hidden_units > 0:
            sequence_output = nn.functional.relu(
                self.slot_hidden(
                    self.dropout(sequence_output)))  # [8, 51, 1536]
            pooled_output = nn.functional.relu(
                self.intent_hidden(self.dropout(pooled_output)))  # [8,1536]

        sequence_output = self.dropout(sequence_output)  # [8, 51, 1536]
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
            # #  处理方式1：mask首位置CLS的mask 置1  否则会报错，后处理忽略掉CLS，TODO
            # # 首位置CLS的mask是0 ，但这样会报错，
            # crf_pred_mask[:, 0] = True
            # # print(crf_pred_mask)
            # crf_seq = self.crf.decode(slot_logits, crf_pred_mask)
            # print(crf_seq)
            # outputs = (crf_seq,)``

            # 1、计算slot_logits后处理取得tag_seq或者此处直接返回tag_seq
            slot_logits = self.slot_classifier(
                sequence_output)[:, 1:, :]  # [8,51,470] -> ([8, 50, 470])
            crf_pred_mask = copy.deepcopy(tag_mask_tensor == 1)[:, 1:]  # copy.deepcopy(tag_mask_tensor == 1) [8,51]
            # print(crf_pred_mask)   # [8,50]
            #  处理方式2：截掉CLS位置从下个位置开始 TODO
            # 首位置CLS的mask是0 ，但这样会报错，

            crf_pred_mask[:, 0] = True
            # print(crf_pred_mask)
            crf_seq = self.crf.decode(slot_logits, crf_pred_mask)
            # print(crf_seq)
            outputs = (crf_seq, )

            # 2、计算intent_logits 后处理取得intent
            pooled_output = self.dropout(pooled_output)
            intent_logits = self.intent_classifier(pooled_output)
            outputs = outputs + (intent_logits, )

            # 3、计算slot损失
            if tag_seq_tensor is not None:
                crf_slot_loss = (-1) * self.crf(slot_logits,
                                                tag_seq_tensor[:, 1:],
                                                crf_pred_mask,
                                                reduction='token_mean')
                outputs = outputs + (crf_slot_loss, )

            # 4、计算intent损失
            if intent_tensor is not None:
                intent_loss = self.intent_loss_fct(intent_logits,
                                                   intent_tensor)
                outputs = outputs + (intent_loss, )
        else:
            # 如果不接CRF
            print("请先使用接CRF，LAST_ADD_CRF=True")
            exit(0)

        # print(outputs)
        # tag_seq_id, intent_logits, (crf_slot_loss), (intent_loss),
        # print("run")
        return outputs


if __name__ == "__main__":
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = JointBERTCRFLSTM(model_conf, device, 470, 58 ,51,60)
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
