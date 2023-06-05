#! /bin/bash
python cross_train.py --config_path config/all_context.json 
cd ../../jointBERT_CRF/diachat/
python cross_train.py --config_path config/all_context.json