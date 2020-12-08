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


class Config(object):
    def __init__(self, path='../'):
        self.path = path
        self.quac_path = f'{path}QuacN/processed/'
        self.quac_dataset_path = f'{path}QuacN/Dataset/'
        self.tokenizer_path = f'{path}QuacN/Tokenizer/'
        self.baseline_dataset_path = f'{path}QuacN/BaselineDataset/'
        self.output_path = f'{path}QuacN/Output/'
        self.save_path = f'{path}QuacN/model/'
        self.log_path = f'{path}QuacN/Summary/'
        self.bert_special_tokens_dict = {'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]'}
        self.edit_pad_id = 0
        self.edit2id = {0:  self.edit_pad_id, 'K': 1, 'I': 2, 'D': 3, 'S': 4}
        self.id2edit = dict([(v, k) for k, v in self.edit2id.items()])
        self.cand_len = 20
        self.K = 1
        self.L = 4
        self.train_fold = 8
        self.sample_fold = 8
        self.batch_size = 8
        self.accumulation_steps = 4
        # self.load_epoch = 10
        self.load_epoch = 10
        self.initial_lr = 5e-5
        self.tune_lr = 1e-5
        self.tune_epoch = 5
        self.max_epoch = 25
        self.rl_epoch = 25
        self.rl_begin_epoch = 10
        self.sampling_strategy = 'RISE'
        self.logger = self.get_logger() if path == '../' else None

    def get_logger(self):
        with open(self.path + 'model/logging.yml', 'r') as f_conf:
            dict_conf = yaml.safe_load(f_conf)
        dictConfig(dict_conf)
        logger_model = logging.getLogger('model')
        return logger_model


default_config = Config('')

