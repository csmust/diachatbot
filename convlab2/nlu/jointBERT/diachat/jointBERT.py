import torch
from torch import nn
from transformers import BertModel


class JointBERT(nn.Module):
    def __init__(self, model_config, device, slot_dim, intent_dim, intent_weight=None):
        super(JointBERT, self).__init__()
        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.device = device
        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.]*intent_dim)

        print(model_config['pretrained_weights'])
        self.bert = BertModel.from_pretrained(model_config['pretrained_weights'])
        # 通常意义的dropout解释为：在训练过程的前向传播中，让每个神经元以一定概率p处于不激活的状态。以达到减少过拟合的效果。
        # 这个操作表示使x每个位置的元素都有一定概率归0，以此来模拟现实生活中的某些频道的数据缺失，以达到数据增强的目的。
        # 防止过拟合
        self.dropout = nn.Dropout(model_config['dropout'])
        self.context = model_config['context']
        self.finetune = model_config['finetune']
        self.context_grad = model_config['context_grad']
        self.hidden_units = model_config['hidden_units']
        if self.hidden_units > 0:
            if self.context:
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
            else:
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
            nn.init.xavier_uniform_(self.intent_hidden.weight)
            nn.init.xavier_uniform_(self.slot_hidden.weight)
        else:
            if self.context:
                self.intent_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.slot_num_labels)
            else:
                self.intent_classifier = nn.Linear(self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.bert.config.hidden_size, self.slot_num_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier.weight)

        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,# 他们四个都是一样的，比如torch.Size([8, 51])
                intent_tensor=None, context_seq_tensor=None, context_mask_tensor=None):#torch.Size([8, 58])  torch.Size([8, 60])  torch.Size([8, 60])
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(input_ids=word_seq_tensor,
                                    attention_mask=word_mask_tensor)
        else:
            outputs = self.bert(input_ids=word_seq_tensor,
                                attention_mask=word_mask_tensor)

        sequence_output = outputs[0]  # [8,51,768]
        pooled_output = outputs[1]  # [8,768]

        if self.context and (context_seq_tensor is not None): #context_seq_tensor [8,60]  context_mask_tensor [8,60]
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            else:
                context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]  #取【1位置】[8,768]
            sequence_output = torch.cat(
                [context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),     #context_output.unsqueeze(1) torch.Size([8, 1, 768]) # .repeat(通道的重复倍数1, 行的重复倍数sequence_output.size(1), 列的重复倍数1) 后torch.Size([8, 51, 768])
                 sequence_output], dim=-1)  # torch.Size([8, 51, 1536])
            pooled_output = torch.cat([context_output, pooled_output], dim=-1)  # [8,1536]

        if self.hidden_units > 0:
            sequence_output = nn.functional.relu(self.slot_hidden(self.dropout(sequence_output)))  # [8, 51, 1536]
            pooled_output = nn.functional.relu(self.intent_hidden(self.dropout(pooled_output)))  #[8,1536]

        sequence_output = self.dropout(sequence_output)   # [8, 51, 1536]
        slot_logits = self.slot_classifier(sequence_output)  # [8,51,470]
        outputs = (slot_logits,)

        pooled_output = self.dropout(pooled_output)  # [8,1536]
        intent_logits = self.intent_classifier(pooled_output)  # [8, 58]
        outputs = outputs + (intent_logits,)

        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1  # [压缩成一个维度 比如408] [True  or  False]
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]  #slot_logits.view(-1, self.slot_num_labels)【408,470】 #active_tag_logits【164,470】
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]  # tag seq tensor 是准确的 tag
            slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)

            outputs = outputs + (slot_loss,)

        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
            outputs = outputs + (intent_loss,)

        return outputs  # slot_logits, intent_logits, (slot_loss), (intent_loss),
