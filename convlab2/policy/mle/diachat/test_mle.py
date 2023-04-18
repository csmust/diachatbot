#encoding=UTF-8
'''
Created on 2022年5月10日

@author: yangjinfeng
'''

import os
import json
import pickle
import zipfile
from convlab2.dst.rule.diachat.dst import RuleDST
from convlab2.policy.mle.diachat.mle import MLE
from convlab2.util.diachat.action_util import *
from copy import deepcopy


def annt_to_actionArry(annotation):
#     actionArry = []
#     for act in annotation:
#         intent = act['act_label']
#         for dsv in act['slot_values']:
#             quad = []
#             quad.append(intent)
#             quad.append(dsv['domain'])
#             quad.append(dsv['slot'])
#             quad.append(dsv['value'])
#             actionArry.append(quad)
    return anntAction_to_actionArry(annotation)

#askfor_askforsure == AskFor or AskForSure
def annt_to_askfor_slots(annotation,askfor_askforsure):
    slotsArry = []
    if askfor_askforsure not in ['AskFor','AskForSure']:
        return slotsArry
    for act in annotation:
        intent = act['act_label']
        if intent == askfor_askforsure:
            for dsv in act['slot_values']:
                quad = []
                quad.append(dsv['domain'])
                quad.append(dsv['slot'])
                quad.append(dsv['value'])
                slotsArry.append(quad)
    return slotsArry
    

def build_test_state():
    part = 'test'
    raw_data = {}
    data = {}
    data[part] = []
    root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    archive = zipfile.ZipFile(os.path.join(root_dir, './data/diachat/{}.json.zip'.format(part)), 'r')
    with archive.open('{}.json'.format(part), 'r') as f:
        raw_data[part] = json.load(f)
                


    dst = RuleDST()
    num=0
    for conversation in raw_data[part]:
        sess = conversation['utterances']
        dst.init_session()
        num+=1

        for i, turn in enumerate(sess):

            domains = turn['domain']

            if domains == '':
                domains = 'none'
                
            sys_action = []   #系统的action，作为训练数据的Y 

            if turn['agentRole'] == 'User':
#                         dst.state['user_domain'] = domains
                dst.state['cur_domain'] = domains
                
                user_action = annt_to_actionArry(turn['annotation'])
                askfor_slots = annt_to_askfor_slots(turn['annotation'],'AskFor')
                askforsure_slotvs = annt_to_askfor_slots(turn['annotation'],'AskForSure')
#                 for act in turn['annotation']:
#                     intent = act['act_label']
#                     intents.append(intent)
#                     for anno in act['slot_values']:
#                         quad = []
#                         quad.append(intent)
#                         quad.append(anno['domain'])
#                         quad.append(anno['slot'])
#                         quad.append(anno['value'])
#                         user_action.append(quad)
                dst.state['askfor_slots'] = askfor_slots
                dst.state['askforsure_slotvs'] = askforsure_slotvs
                dst.state['user_action'] = user_action

                if i + 2 == len(sess):
                    dst.state['terminated'] = True

            else: #agentRole=Doctor
                dst.state['belief_state'] =  turn['sys_state_init']   
                sys_action = annt_to_actionArry(turn['annotation'])
#                 for act in turn['annotation']:
#                     intent = act['act_label']
#                     for anno in act['slot_values']:
#                         quad = []
#                         quad.append(intent)
#                         quad.append(anno['domain'])
#                         quad.append(anno['slot'])
#                         quad.append(anno['value'])
#                         sys_action.append(quad)
                
                test_X = deepcopy(dst.state)   #553
                test_Y = deepcopy(sys_action)           #186
                test_Y_orig = turn['annotation']
                #print("X dim is %d, Y dim is %d" % (len(training_X),len(training_Y))) #X dim is 553, Y dim is 186
                data[part].append([test_X,test_Y,test_Y_orig])
                dst.state['system_action'] = sys_action

    return data


if __name__ == '__main__':
    #simply testing
#     data = build_test_state()
#     mle = MLE()
#     test_X,test_Y,test_Y_orig = data['test'][2]
#     prd_Y = mle.predict(test_X)
#     print("golden: ",test_Y_orig)
#     print("predict: ",prd_Y)
    
    
    data = build_test_state()
    mle = MLE()
 
    policy_output = {}
    seq = 0;
    for test_X,_,test_Y, in data['test']:
        prd_Y = mle.predict(test_X)
        policy_output[str(seq)] = {"state":test_X,"golden_action":test_Y,"prediction":prd_Y}
        seq += 1
 
    output_file = 'policy_output.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(policy_output, f, ensure_ascii=False, sort_keys=True, indent=4)

    