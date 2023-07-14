LAST_ADD_CRF=True
USELSTM=True
CRF_LEARNING_RATE = 1e-2
LSTM_LEARNING_RATE = 1e-3
LSTM_HIDDEN_SIZE = 512
USE_LOSSGATE = False
DROPOUT=0.1
DROPOUTINTENT=0.2
from dca_jointmodel import Joint_model
device="cuda:0"
HIDENSIZE = 128 #768
LSTM_DROPOUT = 0.5
attention_dropout = 0.1
use_gpu=True