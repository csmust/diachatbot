import os
import json
import pickle
import zipfile
import torch
import torch.utils.data as data
# from convlab2.util.diachat.state import default_state
from convlab2.dst.rule.diachat.dst import RuleDST
# from convlab2.policy.vector.vector_diachat import DiachatVector
from copy import deepcopy


class PolicyDataLoaderDiachat():

    def __init__(self,vectoriser):
        root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
#         voc_file = os.path.join(root_dir, 'policy_test/data/diabetes/sys_da_voc.json')
#         voc_opp_file = os.path.join(root_dir, 'policy_test/data/diabetes/usr_da_voc.json')
#         usr_domain_file = os.path.join(root_dir, 'policy_test/data/diabetes/user_domain_da_voc.json')
#         sys_domain_file = os.path.join(root_dir, 'policy_test/data/diabetes/sys_domain_da_voc.json')
#         sys_intent_domain_file = os.path.join(root_dir, 'policy_test/data/diabetes/sys_intent_domain.json')
#         self.vector = DiachatVector(sys_da_voc_json=voc_file, usr_da_voc_json=voc_opp_file, sys_domain_da_voc_json=sys_domain_file, usr_domain_da_voc_json=usr_domain_file,sys_intent_domain_json=sys_intent_domain_file)
        self.vector =  vectoriser
        processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')
        if os.path.exists(processed_dir):
            print('Load processed data file')
            self._load_data(processed_dir)
        else:
            print('Start preprocessing the dataset')
            print(root_dir)
            self._build_data(root_dir, processed_dir)
            
    def setVectoriser(self,vector):
        self.vector =  vector

    def _build_data(self, root_dir, processed_dir):
        raw_data = {}
        #读取这三个部分的数据
        for part in ['train', 'val', 'test']:
            archive = zipfile.ZipFile(os.path.join(root_dir, './data/diachat/{}.json.zip'.format(part)), 'r')
            with archive.open('{}.json'.format(part), 'r') as f:
                raw_data[part] = json.load(f)

        self.data = {}

        dst = RuleDST()
        for part in ['train', 'val', 'test']:
            self.data[part] = []
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
                        user_action = []
                        intents=[]
                        for act in turn['annotation']:
                            intent = act['act_label']
                            intents.append(intent)
                            for anno in act['slot_values']:
                                quad = []
                                quad.append(intent)
                                quad.append(anno['domain'])
                                quad.append(anno['slot'])
                                quad.append(anno['value'])
                                user_action.append(quad)

                        dst.state['user_action'] = user_action

                        if i + 2 == len(sess):
                            dst.state['terminated'] = True

                    else: #agentRole=Doctor
                            #清空belief_state部分
                        
                        dst.state['belief_state'] =  turn['sys_state_init']   
#                         for domain, svs in dst.state['belief_state'].items():
#                             if domain != '基本信息':
#                                 for SEA, entry_list in svs.items():
#                                     del entry_list[1:]
# 
#                         #构建新的belief_state
#                         for domain, svs in turn['sys_state_init'].items():
#                             for Status_Explanation_Advice, slot_value_pair_list in svs.items():
#                                 if Status_Explanation_Advice != '疑问':
#                                     # 基本信息不区分'现状'、'解释'、'建议'，少一层dict，不用deepcopy
#                                     # 每个slot不会有多个元素（例如'体重'，只有一个值），不用建立多个entry
#                                     if domain=='基本信息':
#                                         for slot_value_pair in slot_value_pair_list:
#                                             for slot, value in slot_value_pair.items():
#                                                 if slot != '':
#                                                     dst.state['belief_state'][domain][slot]=value
# 
#                                     # 其他domain区分'现状'、'解释'、'建议'，为了避免覆盖
#                                     # 需要deepcopy，并为每一个组slot-value建立一个entry
#                                     else:
#                                         for slot_value_pair in slot_value_pair_list:
#                                             info = deepcopy(dst.state['belief_state'][domain][Status_Explanation_Advice][0])
#                                             for slot, value in slot_value_pair.items():
#                                                     # if slot != 'selectedResults':
#                                                 if domain!='' and slot != '':
#                                                     info[slot]=value
#                                             dst.state['belief_state'][domain][Status_Explanation_Advice].append(info)

                        sys_action = []
#                         sys_intents=[]
                        for act in turn['annotation']:
                            intent = act['act_label']
#                             sys_intents.append(intent)
                            for anno in act['slot_values']:
                                quad = []
                                quad.append(intent)
                                quad.append(anno['domain'])
                                quad.append(anno['slot'])
                                quad.append(anno['value'])
                                sys_action.append(quad)
                        
                        training_X = self.vector.state_vectorize(deepcopy(dst.state))   #553
                        training_Y = self.vector.action_vectorize(sys_action)           #186
                        #print("X dim is %d, Y dim is %d" % (len(training_X),len(training_Y))) #X dim is 553, Y dim is 186
                        self.data[part].append([training_X,training_Y])
                        dst.state['system_action'] = sys_action

        os.makedirs(processed_dir)
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'wb') as f:
                pickle.dump(self.data[part], f)

    def _load_data(self, processed_dir):
        self.data = {}
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'rb') as f:
                self.data[part] = pickle.load(f)

    def create_dataset(self, part, batchsz):
        print('Start creating {} dataset'.format(part))
        s = []
        a = []
        for item in self.data[part]:
            s.append(torch.Tensor(item[0]))
            a.append(torch.Tensor(item[1]))
        s = torch.stack(s)
        a = torch.stack(a)
        dataset = Dataset(s, a)
        dataloader = data.DataLoader(dataset, batchsz, True)
        print('Finish creating {} dataset'.format(part))
        return dataloader


class Dataset(data.Dataset):
    def __init__(self, s_s, a_s):
        self.s_s = s_s
        self.a_s = a_s
        self.num_total = len(s_s)

    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        return s, a

    def __len__(self):
        return self.num_total

# manager = PolicyDataLoaderDiachat()