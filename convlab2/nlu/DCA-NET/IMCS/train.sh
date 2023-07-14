#! /bin/bash
python preprocess.py bert-base-chinese && python train.py --config_path config/all_context.json && python test.py --config_path config/all_context.json