#! /bin/bash
python cross_train_afteraugment.py --config_path config/all_context.json 
cd ../../jointBERT/diachat/
python cross_train_afteraugment.py --config_path config/all_context.json