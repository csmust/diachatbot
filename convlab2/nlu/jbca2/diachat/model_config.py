LAST_ADD_CRF=True
USELSTM=True
CRF_LEARNING_RATE = 1e-2
LSTM_LEARNING_RATE = 1e-3
ATTENTION_LEARNING_RATE = 3e-5
LSTM_HIDDEN_SIZE = 512
USE_LOSSGATE = False
DROPOUT=0.1

# MUSE_HEAD=1
# MUSE_MASK=True
ATTENTION_MASK=True
from convlab2.nlu.jbca2.diachat.jointBERT_biModle_attention2 import JointBERTCRFLSTM