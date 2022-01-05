#from stsgcn.train_sts import train
#config_path = "../configs/train_stsgcn.yaml"
#from stsgcn.train_rnn import train
#config_path = "../configs/train_rnn.yaml"
from stsgcn.train_rnn_stsE import train
config_path = "../configs/train_rnn_stsE.yaml"

train(config_path)
