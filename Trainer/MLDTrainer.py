import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config_n import Config
import numpy as np
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from Model.BertMLD import BertMLD
config = Config()
logger = config.logger


class MLDTrainer:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.batch_size = self.batch_size if hasattr(self, 'batch_size') else config.batch_size
        self.accumulation_steps = self.accumulation_steps if hasattr(self, 'accumulation_steps') else config.accumulation_steps
        self.model_name = self.model_name if hasattr(self, 'model_name') else 'baseline'
        self.max_epoch = self.max_epoch if hasattr(self, 'max_epoch') else config.max_epoch
        self.initial_lr = self.initial_lr if hasattr(self, 'initial_lr') else config.initial_lr
        self.tune_lr = self.tune_lr if hasattr(self, 'tune_lr') else config.tune_lr
        self.tune_epoch = self.tune_epoch if hasattr(self, 'tune_epoch') else config.tune_epoch
        self.train_size = self.train_size if hasattr(self, 'train_size') else 1e-3
        self.load_epoch = self.load_epoch if hasattr(self, 'load_epoch') else config.load_epoch
        self.model: BertMLD = self.model if hasattr(self, 'model') else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.accumulation_count = 0
        self.optimizer = self.set_optimizer(self.initial_lr)
        model_bp_count = (self.max_epoch * self.train_size) / (self.batch_size * self.accumulation_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=min(int(model_bp_count / 10), 20000),
                                                         num_training_steps=int(model_bp_count) + 100)
        self.save_path = config.save_path + self.model_name
        self.method = 'train'

    def set_save_path(self, save_path=None, model_name=None):
        if save_path is None:
            save_path = config.save_path
        if model_name is None:
            model_name = self.model_name
        self.save_path = save_path + model_name

    def set_optimizer(self, lr):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if
                        p.requires_grad and not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if
                        p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        self.optimizer = optimizer
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.optimizer = self.optimizer
        logger.info('optimizer ')
        for tup in self.optimizer.param_groups:
            logger.info(f'optimizer lr = {tup["lr"]}, params len {len(tup["params"])}')
        return optimizer

    def save_model(self, epoch):
        save_name = self.save_path + f'-{epoch}'
        logger.info(f"model saved in {save_name}")
        torch.save(self.model.state_dict(), save_name)

    def load_model(self, epoch):
        if epoch <= 0:
            return
        save_name = self.save_path + f'-{epoch}'
        if os.path.exists(save_name):
            # print(f'------------------------{save_name}')
            logger.info(f"model load in {save_name}")
            self.model.load_state_dict(torch.load(save_name, map_location=self.device), strict=False)
        else:
            raise FileNotFoundError(f'{save_name} not found')

















