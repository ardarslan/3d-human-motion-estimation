#from stsgcn.train_sts import train
#config_path = "../configs/train_stsgcn.yaml"
from stsgcn.train_rnn import train
config_path = "../configs/train_rnn.yaml"

train(config_path)
