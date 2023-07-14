import argparse
import os
import json
import random
import numpy as np
import torch
from dataloader import Dataloader
from model_config import *
from mylogger import Logger
import sys

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


parser = argparse.ArgumentParser(description="Test a model.")
parser.add_argument('--config_path',
                    help='path to config file')


if __name__ == '__main__':
    args = parser.parse_args()
    config = json.load(open(args.config_path))
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    log_dir = config['log_dir']
    DEVICE = config['DEVICE']

    set_seed(config['seed'])

    print('-' * 20 + 'data' + '-' * 20)
    from postprocess import is_slot_da, calculateF1, recover_intent

    intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json'),encoding='utf-8'))
    tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json'),encoding='utf-8'))
    dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
                            pretrained_weights=config['model']['pretrained_weights'])
    print('intent num:', len(intent_vocab))
    print('tag num:', len(tag_vocab))
    for data_key in ['val', 'test']:
        dataloader.load_data(json.load(open(os.path.join(data_dir, '{}_data.json'.format(data_key)),encoding='utf-8')), data_key,
                             cut_sen_len=config['cut_sen_len'], use_bert_tokenizer=config['use_bert_tokenizer'])
        print('{} set size: {}'.format(data_key, len(dataloader.data[data_key])))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout=Logger(filename=os.path.join(log_dir,'test.log'),stream=sys.stdout)
    model = Joint_model(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim,dataloader.max_sen_len,dataloader.max_context_len)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()

    batch_size = config['model']['batch_size']

    data_key = 'test'
    predict_golden = {'intent': [], 'slot': [], 'overall': []}
    slot_loss, intent_loss = 0, 0
    for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(batch_size, data_key=data_key):
        pad_batch = tuple(t.to(DEVICE) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        if not config['model']['context']:
            context_seq_tensor, context_mask_tensor = None, None

        with torch.no_grad():
            tag_seq_id, intent_logits, batch_slot_loss, batch_intent_loss = model.forward(word_seq_tensor,
                                                                                           word_mask_tensor,
                                                                                           tag_seq_tensor,
                                                                                           tag_mask_tensor,
                                                                                           intent_tensor,
                                                                                           context_seq_tensor,
                                                                                           context_mask_tensor)
        slot_loss += batch_slot_loss.item() * real_batch_size
        intent_loss += batch_intent_loss.item() * real_batch_size
        for j in range(real_batch_size):
            predicts = sorted(recover_intent(dataloader, intent_logits[j], tag_seq_id[j], tag_mask_tensor[j],
                                      ori_batch[j][0], ori_batch[j][-4]))
            labels = sorted(ori_batch[j][3])

            # print("predict:",predicts)
            # print("golden:",labels)
            # print("\n")
            # if "".join(ori_batch[j][0]) in "这个是正常的数值，餐后的两个小时的血糖值在3.9-7.8是正常的":
            #     print("====")
            

            predict_golden['overall'].append({
                # "ori": dataloader.tokenizer.decode(ori_batch[j][4][2:-1]),
                "ori": "".join(ori_batch[j][0]),
                'predict': predicts,
                'golden': labels,
            })
            predict_golden['slot'].append({
                'predict': [x for x in predicts if is_slot_da(x)],
                'golden': [x for x in labels if is_slot_da(x)]
            })
            predict_golden['intent'].append({
                'predict': [x for x in predicts if not is_slot_da(x)],
                'golden': [x for x in labels if not is_slot_da(x)]
            })
        print('[%d|%d] samples' % (len(predict_golden['overall']), len(dataloader.data[data_key])))

    total = len(dataloader.data[data_key])
    slot_loss /= total
    intent_loss /= total
    print('%d samples %s' % (total, data_key))
    print('\t slot loss:', slot_loss)
    print('\t intent loss:', intent_loss)
    
    for x in ['intent', 'slot', 'overall']:
        precision, recall, F1 ,acc= calculateF1(predict_golden[x])
        print('-' * 20 + x + '-' * 20)
        print('\t Precision: %.2f' % (100 * precision))
        print('\t Recall: %.2f' % (100 * recall))
        print('\t F1: %.2f' % (100 * F1))
        print('\t acc: %.2f' % (100 * acc))


    output_file = os.path.join(output_dir, 'output.json')
    json.dump(predict_golden['overall'], open(output_file, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    
    # python脚本示例
    import requests
    headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjExODgzLCJ1dWlkIjoiMTczM2I5ZjYtYjk0YS00OWU4LWI0ZDYtMWFjMjIxZWUzN2Y3IiwiaXNfYWRtaW4iOmZhbHNlLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.ClcRp5unEd_SHlj9NfQgeof5XuOCKA9SzqboYVfrmG_gMkx0xj22Ypdm3_CtoDqyLG-lqEpsWsr7-PFpcMa6xQ"}

    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                     json={
                         "title": "eg. 来自我的程序",
                         "name": f"eg. 我的ImageNet实验",
                         "content": "eg. Epoch=100. Acc=90.2"
                     }, headers = headers)
    print(resp.content.decode())


    
