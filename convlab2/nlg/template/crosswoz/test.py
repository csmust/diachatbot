import os
import json
from convlab2.nlg.template.crosswoz.nlg import TemplateNLG
import convlab2.nlg.template.crosswoz.nlg as nlg
from pathlib import Path
import torch
template_dir = os.path.dirname(nlg.__file__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# template = read_json(os.path.join(template_dir, 'auto_user_template_nlg.json'))    
template = read_json(os.path.join(template_dir, 'auto_system_template_nlg.json'))
# template = read_json(os.path.join(template_dir, 'manual_user_template_nlg.json'))
# for key,values in template.items():
#     print("%4d,%s" % (len(values),key))


print(Path(Path.home()))  #C:\Users\admin

print(torch.__version__)
#'1.10.2+cu113'
print(torch.cuda.is_available())
#True
print(torch.cuda.device_count())
#1
print(torch.cuda.get_device_name())