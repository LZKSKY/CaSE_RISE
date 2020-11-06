import os
import torch
import logging
from logging.config import dictConfig
import yaml
import sys
import random
import numpy as np
seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.set_num_threads(4)
sys.path.append(__file__)
# print(sys.path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# path = '../../'
path = '../'

quac_path = f'{path}QuacN/processed/'
quac_dataset_path = f'{path}QuacN/Dataset/'
baseline_dataset_path = f'{path}QuacN/BaselineDataset/'
output_path = f'{path}QuacN/Output/'
save_path = f'{path}QuacN/model/'


# if not os.path.exists(quac_dataset_path):
#     os.mkdir(quac_dataset_path)
#
# if not os.path.exists(baseline_dataset_path):
#     os.mkdir(baseline_dataset_path)

lif_name = 'lif'
gpt2_special_tokens_dict = {'sep_token': '[SEP]', 'pad_token': '[PAD]', 'bos_token': '[BOS]', 'eos_token': '[EOS]', 'cls_token': '[CLS]'}
bert_special_tokens_dict = {'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]'}

cand_len = 20

if path == '../':
    with open(path + 'model/logging.yml', 'r') as f_conf:
        dict_conf = yaml.safe_load(f_conf)
    dictConfig(dict_conf)
    logger_model = logging.getLogger('model')
else:
    logger_model = None


