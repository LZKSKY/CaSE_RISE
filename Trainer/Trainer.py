import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from time import time
from Trainer.common import EMA, EarlyStopping
from config import logger_model as logger
from config import save_path
import numpy as np
import os
from transformers import AdamW, get_linear_schedule_with_warmup


class Trainer:
    def __init__(self, model, batch_size=32, accumulation_steps=1, model_name='baseline',
                 ema_rate=0.995, max_epoch=50, initial_lr=1e-3, tune_lr=1e-5, tune_epoch=20, train_size=1e3):
        self.accumulation_count = 0
        self.accumulation_steps = accumulation_steps
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.batch_size = batch_size
        self.model_name = model_name
        self.begin_epoch = -1
        self.tune_lr = tune_lr
        self.initial_lr = initial_lr
        self.tune_epoch = tune_epoch
        self.max_epoch = max_epoch
        self.optimizer = self.set_init_optimizer(lr=self.initial_lr)
        model_bp_count = (max_epoch * train_size) / (batch_size * accumulation_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=min(int(model_bp_count / 10), 2000), num_training_steps=int(model_bp_count) + 100)
        # self.ema = EMA(self.model, ema_rate)
        # self.ema.register()
        self.save_path = save_path + self.model_name
        self.earlystopping = EarlyStopping(patience=5, verbose=False, delta=0, path=self.save_path + '-checkpoint.pt',
                                           trace_func=logger.info)
        if self.optimizer:
            logger.info(f'optimizer lr = {self.optimizer.param_groups[0]["lr"]}, params len {len(self.optimizer.param_groups[0]["params"])}')

    def set_init_optimizer(self, lr):
        # for name, para in self.model.named_parameters():
        #     para.requires_grad = False
            # if name[0] != 'h':
            #     para.requires_grad = True
            # else:
            #     para.requires_grad = False
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # params = filter(lambda x: x.requires_grad, self.model.parameters())
        # print('-----------------------------------------------------')
        # print(len(list(params)))
        # print(len(optimizer_grouped_parameters[0]['params']))
        # print(len(optimizer_grouped_parameters[1]['params']))
        # print('-----------------------------------------------------')
        return AdamW(optimizer_grouped_parameters, lr=lr)

    def set_optimizer(self, lr, params):
        self.optimizer = AdamW(params, lr=lr)
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=2000,
        #                                                  num_training_steps=int(model_bp_count) + 100)
        if self.scheduler is not None:
            self.scheduler.optimizer = self.optimizer
        logger.info(f'optimizer lr = {self.optimizer.param_groups[0]["lr"]}, params len {len(self.optimizer.param_groups[0]["params"])}')

    def train_batch(self, batch_data):
        self.accumulation_count += 1
        output, loss = self.model(batch_data, method='train')
        if isinstance(loss, tuple) or isinstance(loss, list):
            closs = [l.mean().cpu().item() for l in loss]
            loss = torch.cat([l.mean().reshape(1) for l in loss]).sum()
        else:
            loss = loss.mean()
            closs = [loss.cpu().item()]

        loss = loss / self.accumulation_steps
        loss.backward()

        if self.accumulation_count % self.accumulation_steps == 0:
            clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            # self.ema.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        return closs

    def train_epoch(self, train_dataset, train_collate_fn, epoch=-1):
        self.model.train()
        train_loader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size,
                                  shuffle=True, pin_memory=False)
        loss_arr = []
        for step, batch_data in tqdm(enumerate(train_loader)):
            start_time = time()
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = batch_data[key].to(self.device)
            loss = self.train_batch(batch_data)
            loss_arr.append(loss)
            if step > 0 and step % 100 == 0:
                elapsed_time = time() - start_time
                info = ['Method', self.model_name, 'Epoch', epoch, 'Batch ', step, 'Loss ',
                        loss, 'Time ', elapsed_time]
                if self.scheduler is not None:
                    info.extend(['Learning rate ', self.scheduler.get_last_lr()])
                logger.info([' '.join(map(lambda x:str(x), info))])
        loss = np.mean(np.concatenate(loss_arr, axis=0), axis=0)
        return loss

    def metrics(self, output, batch_data):
        pass

    def eval_batch(self, batch_data):
        output = self.model(batch_data, method='eval')
        return output

    def eval_epoch(self, eval_dataset, eval_collate_fn, epoch=-1):
        self.model.eval()
        eval_loader = DataLoader(eval_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size * 2,
                                 shuffle=True, pin_memory=False)
        start_time = time()
        metrics_arr = []
        for step, batch_data in tqdm(enumerate(eval_loader)):
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = batch_data[key].to(self.device)
            metrics = self.eval_batch(batch_data)
            metrics_arr.append(metrics)
        elapsed_time = time() - start_time
        # metrics = np.mean(np.concatenate(metrics_arr, axis=0), axis=0)
        metrics = np.mean(metrics_arr)
        info = ['Method', self.model_name, 'Epoch', epoch, 'Loss ', metrics, 'Time ', elapsed_time]
        logger.info([' '.join(map(lambda x: str(x), info))])
        return metrics

    def save_model(self, epoch):
        save_name = self.save_path + f'-{epoch}'
        logger.info(f"model saved in {save_name}")
        torch.save(self.model.state_dict(), save_name)

    def load_model(self, epoch):
        save_name = self.save_path + f'-{epoch}'
        if os.path.exists(save_name):
            self.begin_epoch = epoch
            logger.info(f"model load in {save_name}")
            self.model.load_state_dict(torch.load(save_name, map_location=self.device))
        else:
            logger.info(f"Not find {save_name}")

    def train(self, train_dataset, train_collate_fn, eval_dataset, eval_collate_fn, max_epoch=25):
        for epoch in range(self.begin_epoch + 1,  max_epoch):
            if self.tune_epoch > 0 and epoch == self.tune_epoch:
                for para in self.model.parameters():
                    para.requires_grad = True
                self.set_optimizer(self.tune_lr, self.model.parameters())
            loss = self.train_epoch(train_dataset, train_collate_fn, epoch)
            metrics = self.eval_epoch(eval_dataset, eval_collate_fn, epoch)
            self.save_model(epoch)






















