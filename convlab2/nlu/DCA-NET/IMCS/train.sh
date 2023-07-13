#! /bin/bash
python preprocess.py hfl/chinese-macbert-large && python train.py --config_path config/all_context.json 