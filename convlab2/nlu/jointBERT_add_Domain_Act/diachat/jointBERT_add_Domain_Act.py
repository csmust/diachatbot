import torch
from torch import nn
from transformers import BertModel

class JointBERT_add_Domain_Act(nn.Module):
    def __init__(self, model_config, device, slot_dim, intent_dim,domain_dim,act_dim ,intent_weight=None,domain_weight=None,act_weight=None):
        super(JointBERT_add_Domain_Act, self).__init__()
        self.slot_num_labels = slot_dim
        self.intent_num_labels = intent_dim
        self.domain_num_labels=domain_dim
        self.act_num_labels=act_dim

        self.device = device
        self.intent_weight = intent_weight if intent_weight is not None else torch.tensor([1.]*intent_dim)
        self.domain_weight = domain_weight if domain_weight is not None else torch.tensor([1.]*domain_dim)
        self.act_weight = act_weight if act_weight is not None else torch.tensor([1.]*act_dim)

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

                self.domain_classifier = nn.Linear(self.hidden_units, self.domain_num_labels)
                self.domain_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
                self.act_classifier = nn.Linear(self.hidden_units, self.act_num_labels)
                self.act_hidden = nn.Linear(2 * self.bert.config.hidden_size, self.hidden_units)
            else:
                self.intent_classifier = nn.Linear(self.hidden_units, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.hidden_units, self.slot_num_labels)
                self.intent_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
                self.slot_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)

                self.domain_classifier = nn.Linear(self.hidden_units, self.domain_num_labels)
                self.domain_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
                self.act_classifier = nn.Linear(self.hidden_units, self.act_num_labels)
                self.act_hidden = nn.Linear(self.bert.config.hidden_size, self.hidden_units)
            nn.init.xavier_uniform_(self.intent_hidden.weight)
            nn.init.xavier_uniform_(self.slot_hidden.weight)
            nn.init.xavier_uniform_(self.domain_hidden.weight)
            nn.init.xavier_uniform_(self.act_hidden.weight)
        else:
            if self.context:
                self.intent_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.slot_num_labels)
                self.domain_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.domain_num_labels)
                self.act_classifier = nn.Linear(2 * self.bert.config.hidden_size, self.act_num_labels)
            else:
                self.intent_classifier = nn.Linear(self.bert.config.hidden_size, self.intent_num_labels)
                self.slot_classifier = nn.Linear(self.bert.config.hidden_size, self.slot_num_labels)
                self.domain_classifier = nn.Linear(self.bert.config.hidden_size, self.domain_num_labels)
                self.act_classifier = nn.Linear(self.bert.config.hidden_size, self.act_num_labels)
        nn.init.xavier_uniform_(self.intent_classifier.weight)
        nn.init.xavier_uniform_(self.slot_classifier.weight)
        nn.init.xavier_uniform_(self.domain_classifier.weight)
        nn.init.xavier_uniform_(self.act_classifier.weight)

        self.intent_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()
        self.domain_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.domain_weight)
        self.act_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.act_weight)

    def forward(self, word_seq_tensor, word_mask_tensor, tag_seq_tensor=None, tag_mask_tensor=None,
                intent_tensor=None, domain_tensor=None,act_tensor=None,context_seq_tensor=None, context_mask_tensor=None):
        """
            return: slot_logits, intent_logits,domain_logits,act_logits ,(slot_loss), (intent_loss), (domain_loss),(act_loss)
        """
        if not self.finetune:
            self.bert.eval()
            with torch.no_grad():
                outputs = self.bert(input_ids=word_seq_tensor,
                                    attention_mask=word_mask_tensor)
        else:
            outputs = self.bert(input_ids=word_seq_tensor,
                                attention_mask=word_mask_tensor)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        if self.context and (context_seq_tensor is not None):
            if not self.finetune or not self.context_grad:
                with torch.no_grad():
                    context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            else:
                context_output = self.bert(input_ids=context_seq_tensor, attention_mask=context_mask_tensor)[1]
            sequence_output = torch.cat(
                [context_output.unsqueeze(1).repeat(1, sequence_output.size(1), 1),
                 sequence_output], dim=-1)
            pooled_output = torch.cat([context_output, pooled_output], dim=-1)

        if self.hidden_units > 0:
            sequence_output = nn.functional.relu(self.slot_hidden(self.dropout(sequence_output)))
            pooled_output_intent = nn.functional.relu(self.intent_hidden(self.dropout(pooled_output)))
            pooled_output_domain = nn.functional.relu(self.domain_hidden(self.dropout(pooled_output)))
            pooled_output_act = nn.functional.relu(self.act_hidden(self.dropout(pooled_output)))

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        outputs = (slot_logits,)

        pooled_output_intent = self.dropout(pooled_output_intent)
        intent_logits = self.intent_classifier(pooled_output_intent)
        outputs = outputs + (intent_logits,)

        
        pooled_output_domain = self.dropout(pooled_output_domain)
        domain_logits = self.domain_classifier(pooled_output_domain)
        outputs = outputs + (domain_logits,)

        
        pooled_output_act = self.dropout(pooled_output_act)
        act_logits = self.act_classifier(pooled_output_act)
        outputs = outputs + (act_logits,)

        if tag_seq_tensor is not None:
            active_tag_loss = tag_mask_tensor.view(-1) == 1
            active_tag_logits = slot_logits.view(-1, self.slot_num_labels)[active_tag_loss]
            active_tag_labels = tag_seq_tensor.view(-1)[active_tag_loss]
            slot_loss = self.slot_loss_fct(active_tag_logits, active_tag_labels)

            outputs = outputs + (slot_loss,)

        if intent_tensor is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_tensor)
            outputs = outputs + (intent_loss,)
        
        if domain_tensor is not None:
            domain_loss = self.domain_loss_fct(domain_logits, domain_tensor)
            outputs = outputs + (domain_loss,)
        
        if act_tensor is not None:
            act_loss = self.act_loss_fct(act_logits, act_tensor)
            outputs = outputs + (act_loss,)

        

        return outputs  # slot_logits, intent_logits,domain_logits,act_logits ,(slot_loss), (intent_loss), (domain_loss),(act_loss)