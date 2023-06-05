import argparse
from asyncio.log import logger
from math import sqrt
import os
import sys
import json
import time

from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import zipfile
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from convlab2.nlu.jointBERT_CRF_add_Act_add_lossgate.diachat.dataloader import Dataloader
from convlab2.nlu.jointBERT_CRF_add_Act_add_lossgate.diachat.jointBERT_CRF_add_Act import JointBERT_CRF_add_Act
from datetime import datetime
from mylogger import Logger

from model_config import *
import torch 
import  torch.nn.functional as F

cross_best_f1=0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def train(CROSS_TRAIN=False,best_val_F1_list=[],args=None):

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    config = json.load(open(args.config_path))
    data_dir = config['data_dir']
    # output_dir = os.path.join(config['output_dir'],TIMESTAMP)   #正式训练时候分别保存模型，以便对比
    output_dir = config['output_dir'] 
    
    log_dir=os.path.join(config['log_dir'],TIMESTAMP)
    DEVICE = config['DEVICE']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout=Logger(filename=os.path.join(log_dir,'train.log'),stream=sys.stdout)
    print(config["model"])

    set_seed(config['seed'])

    print('-' * 20 + 'data' + '-' * 20)
    from convlab2.nlu.jointBERT_CRF_add_Act_add_lossgate.diachat.postprocess import is_slot_da, calculateF1, recover_intent

    intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json'),encoding='utf-8'))
    tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json'),encoding='utf-8'))
    domain_vocab = json.load(open(os.path.join(data_dir, 'domain_vocab.json'), encoding='utf-8'))
    act_vocab = json.load(open(os.path.join(data_dir, 'act_vocab.json'), encoding='utf-8'))
    
    
    
    dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
                            domain_vocab=domain_vocab, act_vocab=act_vocab,
                            pretrained_weights=config['model']['pretrained_weights'])
    
    print('intent num:', len(intent_vocab))
    print('tag num:', len(tag_vocab))
    print("act num:",len(act_vocab))
    if CROSS_TRAIN:
         for data_key in ['train', 'val']:
            dataloader.load_data(json.load(open(os.path.join(data_dir, '{}_data.json'.format(data_key)),encoding='utf-8')), data_key,
                                cut_sen_len=config['cut_sen_len'], use_bert_tokenizer=config['use_bert_tokenizer'])
            print('{} set size: {}'.format(data_key, len(dataloader.data[data_key])))
    else:
        for data_key in ['train', 'val', 'test']:
            dataloader.load_data(json.load(open(os.path.join(data_dir, '{}_data.json'.format(data_key)),encoding='utf-8')), data_key,
                                cut_sen_len=config['cut_sen_len'], use_bert_tokenizer=config['use_bert_tokenizer'])
            print('{} set size: {}'.format(data_key, len(dataloader.data[data_key])))
        


    writer = SummaryWriter(log_dir)
    model = JointBERT_CRF_add_Act(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim,
    dataloader.domain_dim, dataloader.act_dim, dataloader.intent_weight,dataloader.domain_weight, dataloader.act_weight)

    # for name, para in  model.named_parameters():
    #     '''打印模型结构'''
    #     print(name)
    #     print("是否被训练: ",para.requires_grad)
    model.to(DEVICE)
    #TODO 参考BERT NER 分别设置CRF和bert学习率
    if config['model']['finetune']:
        no_decay = ['bias', 'LayerNorm.weight']
        crf = ["crf"]
        no_decay_and_crf =no_decay + crf
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay_and_crf) and  p.requires_grad],
             'weight_decay': config['model']['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in crf) and p.requires_grad],
             'weight_decay': config['model']['weight_decay'],"lr":CRF_LEARNING_RATE},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config['model']['learning_rate'],
                          eps=config['model']['adam_epsilon'])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['model']['warmup_steps'],
                                                    num_training_steps=config['model']['max_step'])
    else:
        for n, p in model.named_parameters():
            if 'bert' in n:
                p.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=config['model']['learning_rate'])

    max_step = config['model']['max_step']
    check_step = config['model']['check_step']
    batch_size = config['model']['batch_size']
    model.zero_grad()
    train_slot_loss, train_intent_loss = 0, 0
    train_act_loss = 0
    best_val_f1 = 0.

    writer.add_text('config', json.dumps(config))
    stime = time.time()
    alpha = 1.0
    beta = 1.0
    lr=config['model']['learning_rate']
    for step in range(1, max_step + 1):
        # start_time = time.time()
        model.train()
        batched_data = dataloader.get_train_batch(batch_size)
        batched_data = tuple(t.to(DEVICE) for t in batched_data)
        # word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = batched_data
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor, domain_tensor, act_tensor = batched_data
        if not config['model']['context']:
            context_seq_tensor, context_mask_tensor = None, None
        _, _, _, slot_loss, intent_loss ,act_loss= model.forward(word_seq_tensor, word_mask_tensor, tag_seq_tensor, tag_mask_tensor,
                                                     intent_tensor, domain_tensor, act_tensor, context_seq_tensor, context_mask_tensor)
        train_slot_loss += slot_loss.item()
        train_intent_loss += intent_loss.item()
        train_act_loss += act_loss.item()

        
        loss = alpha  * slot_loss + beta * intent_loss + (3-alpha-beta) * act_loss
        loss.backward()
        # alpha = alpha - lr * (2*slot_loss.item()-intent_loss.item()-act_loss.item())
        # beta = beta - lr * (2*intent_loss.item()-slot_loss.item()-act_loss.item())
        tmp = torch.Tensor([slot_loss,intent_loss,act_loss])/ (batch_size ** 0.5)
        tmpl=F.softmax(tmp) 
        alpha = alpha - lr * (2*tmpl[0].item() - tmpl[1].item()-tmpl[2].item())
        beta = beta - lr * (2*tmpl[1].item() - tmpl[0].item()-tmpl[2].item())



        # end_time = time.time()
        # print("end_time - start_time",end_time - start_time)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if config['model']['finetune']:
            scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        if step % check_step == 0:
            train_slot_loss = train_slot_loss / check_step
            train_intent_loss = train_intent_loss / check_step
            train_act_loss = train_act_loss / check_step
            print('alpha=',alpha)
            print('beta=',beta)
            print('[%d|%d] step' % (step, max_step))
            print('\t slot loss:', train_slot_loss)
            print('\t intent loss:', train_intent_loss)
            print('\t train act loss:', train_act_loss)
            #by yangjinfeng
            etime = time.time()
            print('%d step  consume time %f'%(step,etime-stime))
            stime = etime

            predict_golden = {'intent': [], 'slot': [], 'overall': []}

            val_slot_loss, val_intent_loss = 0, 0
            val_act_loss = 0
            model.eval()
            for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(batch_size, data_key='val'):
                pad_batch = tuple(t.to(DEVICE) for t in pad_batch)
                word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor,domain_tensor,act_tensor = pad_batch
                if not config['model']['context']:
                    context_seq_tensor, context_mask_tensor = None, None
                with torch.no_grad():
                    tag_seq_id, intent_logits,act_logits, slot_loss, intent_loss,act_loss = model.forward(word_seq_tensor,
                                                                                       word_mask_tensor,
                                                                                       tag_seq_tensor,
                                                                                       tag_mask_tensor,
                                                                                       intent_tensor,
                                                                                       domain_tensor,
                                                                                       act_tensor,
                                                                                       context_seq_tensor,
                                                                                       context_mask_tensor)
                val_slot_loss += slot_loss.item() * real_batch_size
                val_intent_loss += intent_loss.item() * real_batch_size
                val_act_loss += act_loss.item() * real_batch_size
                for j in range(real_batch_size):
                    predicts = recover_intent(dataloader, intent_logits[j], tag_seq_id[j], tag_mask_tensor[j],
                                              ori_batch[j][0], ori_batch[j][-4])
                    labels = ori_batch[j][3]

                    predict_golden['overall'].append({
                        'predict': predicts,
                        'golden': labels
                    })

                    predict_golden['slot'].append({
                        'predict': [x for x in predicts if is_slot_da(x)],
                        'golden': [x for x in labels if is_slot_da(x)]
                    })

                    predict_golden['intent'].append({
                        'predict': [x for x in predicts if not is_slot_da(x)],
                        'golden': [x for x in labels if not is_slot_da(x)]
                    })
            for j in range(10):
                writer.add_text('val_sample_{}'.format(j),
                                json.dumps(predict_golden['overall'][j], indent=2, ensure_ascii=False),
                                global_step=step)
            total = len(dataloader.data['val'])
            val_slot_loss /= total
            val_intent_loss /= total
            val_act_loss /=total
            print('%d samples val' % total)
            print('\t slot loss:', val_slot_loss)
            print('\t intent loss:', val_intent_loss)

            print('\t val act loss:', val_act_loss)

            writer.add_scalar('intent_loss/train', train_intent_loss, global_step=step)
            writer.add_scalar('intent_loss/val', val_intent_loss, global_step=step)

            writer.add_scalar('slot_loss/train', train_slot_loss, global_step=step)
            writer.add_scalar('slot_loss/val', val_slot_loss, global_step=step)

            writer.add_scalar('act_loss/train',
                              train_act_loss, global_step=step)
            writer.add_scalar('act_loss/val',
                              val_act_loss, global_step=step)
            for x in ['intent', 'slot', 'overall']:
                precision, recall, F1 = calculateF1(predict_golden[x])
                print('-' * 20 + x + '-' * 20)
                print('\t Precision: %.2f' % (100 * precision))
                print('\t Recall: %.2f' % (100 * recall))
                print('\t F1: %.2f' % (100 * F1))

                writer.add_scalar('val_{}/precision'.format(x), precision, global_step=step)
                writer.add_scalar('val_{}/recall'.format(x), recall, global_step=step)
                writer.add_scalar('val_{}/F1'.format(x), F1, global_step=step)

            if F1 > best_val_f1:
                best_val_f1 = F1
                torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                print('best val F1 %.4f' % best_val_f1)
                print('save on', output_dir)

            train_slot_loss, train_intent_loss = 0, 0
            train_act_loss = 0
    writer.add_text('val overall F1', '%.2f' % (100 * best_val_f1))
    writer.close()
    print("best_val_f1 = ", best_val_f1)
    
    global cross_best_f1
    if CROSS_TRAIN==True:
        if cross_best_f1 < best_val_f1:
            cross_best_f1=best_val_f1
            print(f"此折 best_val_f1({best_val_f1}) 比历史折大 ")
            model_path = os.path.join(output_dir, 'pytorch_model.bin')

            zip_path = config['zipped_model_path']
            print('best K-fold Cross Validation zip model to', zip_path)
        
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(model_path)

        best_val_F1_list.append(best_val_f1)
        print("best_val_F1_list=",best_val_F1_list)

        return best_val_F1_list
        
    
    else:
        model_path = os.path.join(output_dir, 'pytorch_model.bin')
        zip_path = config['zipped_model_path']
        print('zip model to', zip_path)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(model_path)





#--config_path config\all.json  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument('--config_path',
                    help='path to config file')
    args = parser.parse_args()

    # train(args=args)
    train(args=args,CROSS_TRAIN=True)