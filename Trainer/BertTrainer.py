from Trainer.Trainer import Trainer
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from time import time
from Trainer.common import EMA, EarlyStopping
from config import logger_model as logger
from config import save_path
import numpy as np
from typing import *


def rank_eval_metrics(output):
    from sklearn.metrics import f1_score
    rank_out, rank_label = [], []
    for obj in output:
        rank_out.append(obj['output'])
        rank_label.append(obj['label'])
    rank_out = np.concatenate(rank_out, axis=0)
    rank_label = np.concatenate(rank_label, axis=0)
    score = f1_score(rank_out, rank_label, average='macro')
    return score


def edit_eval_metrics(output, *args):
    if len(args) > 0:
        key1 = args[0]
        key2 = args[1]
    else:
        key1 = 'edit_output'
        key2 = 'edit_label'
    edit_out, edit_label = [], []
    for obj in output:
        edit_out.append(obj[key1])
        edit_label.append(obj[key2])
    edit_out = np.concatenate(edit_out, axis=0)
    edit_label = np.concatenate(edit_label, axis=0)
    edit_slice = edit_label > 0
    acc = np.argmax(edit_out[edit_slice], axis=1) == edit_label[edit_slice]
    acc = np.mean(acc)
    return acc


def gen_eval_metrics(output, *args):
    key1 = args[0] if len(args) > 0 else 'gen_loss'
    out_arr = []
    for obj in output:
        out_arr.append(obj[key1])
    return -np.mean(out_arr)


def edit_gen_eval_metrics(output):
    gen_score = gen_eval_metrics(output)
    if 'edit_loss' in output[0]:
        edit_score = gen_eval_metrics(output, 'edit_loss')
    elif 'edit_out' in output[0]:
        edit_score = edit_eval_metrics(output)
    else:
        edit_score = 0.
    score_arr = [gen_score, edit_score]
    if 'decoder_edit_loss' in output[0]:
        decoder_edit_score = gen_eval_metrics(output, 'decoder_edit_loss')
        score_arr.append(decoder_edit_score)
    elif 'decoder_edit' in output[0]:
        decoder_edit_score = edit_eval_metrics(output, 'decoder_edit', 'decoder_edit_label')
        score_arr.append(decoder_edit_score)

    if 'anaphora_loss' in output[0]:
        anaphora_score = gen_eval_metrics(output, 'anaphora_loss')
        score_arr.append(anaphora_score)
    return score_arr


class BertTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_type = 'rank'
        self.func_map_dict = {'rank': self.model.do_rank, 'r': self.model.do_rank,
                              # 'edit': self.model.do_edit_pred, 'e': self.model.do_edit_pred,
                              # 'gen': self.model.do_gen, 'g': self.model.do_gen,
                              'edit_gen': self.model.do_edit_gen_raw, 'eg': self.model.do_edit_gen_raw}
        self.eval_func_map_dict = {'rank': rank_eval_metrics, 'r': rank_eval_metrics,
                                   'edit_gen': edit_gen_eval_metrics, 'eg': edit_gen_eval_metrics}

    def map_func(self):
        return self.func_map_dict.get(self.train_type, self.model.forward)

    def eval_batch(self, batch_data):
        if isinstance(batch_data, (tuple, list)):
            batch_data, train_func = batch_data
        else:
            train_func = self.model.forward
        output = train_func(batch_data, method='eval')
        if isinstance(output, tuple) or isinstance(output, list):
            output = [l.detach().cpu().numpy() for l in output]
        elif isinstance(output, dict):
            for k in output:
                output[k] = output[k].detach().cpu().numpy()
        else:
            output = [output.detach().cpu().numpy()]
        return output

    def eval_epoch(self, eval_dataset, eval_collate_fn, epoch=-1):
        self.model.eval()
        with torch.no_grad():
            eval_loader = DataLoader(eval_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size * 2,
                                     shuffle=True, pin_memory=False)
            start_time = time()
            metrics_arr = []
            for step, batch_data in tqdm(enumerate(eval_loader)):
                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor):
                        batch_data[key] = batch_data[key].to(self.device)
                metrics = self.eval_batch((batch_data, self.map_func()))
                metrics_arr.append(metrics)
            elapsed_time = time() - start_time
            # metrics = np.mean(np.concatenate(metrics_arr, axis=0), axis=0)
            score = self.eval_func_map_dict[self.train_type](metrics_arr)
            # metrics = np.mean(metrics_arr, axis=0)
            info = ['Method', self.model_name, 'Epoch', epoch, 'Score ', score, 'Time ', elapsed_time]
            logger.info(' '.join(map(lambda x: str(x), info)))
            return np.sum(score)

    def train_batch(self, batch_data):
        self.accumulation_count += 1
        if isinstance(batch_data, (tuple, list)):
            batch_data, train_func = batch_data
        else:
            train_func = self.model.forward
        output, loss = train_func(batch_data, method='train')
        if isinstance(loss, tuple) or isinstance(loss, list):
            closs = [l.mean().cpu().item() for l in loss]
            loss = torch.cat([l.mean().reshape(1) for l in loss]).sum()
        else:
            loss = loss.sum()
            closs = [loss.cpu().item()]
        # print(closs)

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
        start_time = time()
        for step, batch_data in tqdm(enumerate(train_loader)):
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = batch_data[key].to(self.device)
            loss = self.train_batch((batch_data, self.map_func()))
            loss_arr.append(loss)
            if step > 0 and step % 10000 == 0:
                elapsed_time = time() - start_time
                loss = np.mean(np.stack(loss_arr, axis=0), axis=0)
                info = ['Method', self.model_name + '-' + self.train_type, 'Epoch', epoch, 'Batch ', step, 'Loss ',
                        loss, 'Time ', elapsed_time]
                if self.scheduler is not None:
                    info.extend(['Learning rate ', self.scheduler.get_last_lr()])
                logger.info(' '.join(map(lambda x: str(x), info)))
        loss = np.mean(np.stack(loss_arr, axis=0), axis=0)
        info = ['Method', self.model_name + '-' + self.train_type, 'Epoch', epoch, 'All ', 'Loss ', loss]
        logger.info(' '.join(map(lambda x: str(x), info)))
        return loss

    def train(self, train_dataset, train_collate_fn, eval_dataset, eval_collate_fn, max_epoch=25):
        logger.info(f"model_name {self.model_name} begin epoch {self.begin_epoch + 1}, max epoch {max_epoch}")
        for epoch in range(self.begin_epoch + 1, self.max_epoch):
            if self.tune_epoch > 0 and epoch == self.tune_epoch:
                for para in self.model.parameters():
                    para.requires_grad = True
                self.set_optimizer(self.tune_lr, self.model.parameters())
            loss = self.train_epoch(train_dataset, train_collate_fn, epoch)
            score = self.eval_epoch(eval_dataset, eval_collate_fn, epoch)
            if epoch >= self.tune_epoch:
                self.earlystopping(score, self.model)
            self.save_model(epoch)
            # if self.earlystopping.early_stop:
            #     break

    def check_loss(self, train_arr: Dict):
        key_arr = ['rank', 'edit_gen']
        logger.info(f"model_name {self.model_name} begin epoch {self.begin_epoch + 1}, max epoch {self.max_epoch}")
        for epoch in range(self.begin_epoch + 1, self.max_epoch):
            self.load_model(epoch)

            t = time()
            metrics_arr = []
            for k in key_arr:
                self.train_type = k
                self.batch_size = train_arr['batch_size'][k]
                self.accumulation_steps = train_arr['accumulation'][k]
                metrics = self.eval_epoch(train_arr['train_dataset'][k], train_arr['train_collate'][k], epoch)
                metrics_arr.append(metrics)
            score = np.sum(metrics_arr)
            info = ['Method', self.model_name + 'train-Evaluation', 'Epoch', epoch, 'metrics_arr', metrics_arr,
                    'Score ', score, 'Time ', time() - t]
            logger.info(' '.join(map(lambda x: str(x), info)))

            t = time()
            metrics_arr = []
            for k in key_arr:
                self.train_type = k
                self.batch_size = train_arr['batch_size'][k]
                self.accumulation_steps = train_arr['accumulation'][k]
                metrics = self.eval_epoch(train_arr['dev_dataset'][k], train_arr['dev_collate'][k], epoch)
                metrics_arr.append(metrics)
            score = np.sum(metrics_arr)
            info = ['Method', self.model_name + 'Evaluation', 'Epoch', epoch, 'metrics_arr', metrics_arr,
                    'Score ', score, 'Time ', time() - t]
            logger.info(' '.join(map(lambda x: str(x), info)))
            # self.earlystopping(score, self.model)
            # self.save_model(epoch)

    # def train_case(self, train_dataset: Dict, train_collate_fn: Dict, dev_dataset: Dict, dev_collate_fn: Dict, max_epoch=25):
    def train_bert(self, train_arr: Dict):
        # key_arr = ['rank', 'edit_gen']
        key_arr = ['edit_gen', 'rank']
        logger.info(f"model_name {self.model_name} begin epoch {self.begin_epoch + 1}, max epoch {self.max_epoch}")
        for epoch in range(self.begin_epoch + 1, self.max_epoch):
            if self.tune_epoch > 0 and epoch == self.tune_epoch:
                for para in self.model.parameters():
                    para.requires_grad = True
                self.set_optimizer(self.tune_lr, self.model.parameters())
            t = time()
            for k in key_arr:
                self.train_type = k
                self.batch_size = train_arr['batch_size'][k]
                self.accumulation_steps = train_arr['accumulation'][k]
                loss = self.train_epoch(train_arr['train_dataset'][k], train_arr['train_collate'][k], epoch)
            metrics_arr = []
            for k in key_arr:
                self.train_type = k
                self.batch_size = train_arr['batch_size'][k]
                self.accumulation_steps = train_arr['accumulation'][k]
                metrics = self.eval_epoch(train_arr['dev_dataset'][k], train_arr['dev_collate'][k], epoch)
                metrics_arr.append(metrics)
            score = np.sum(metrics_arr)
            info = ['Method', self.model_name + '-Evaluation', 'Epoch', epoch, 'Score ',
                    score, 'Time ', time() - t]
            logger.info(' '.join(map(lambda x: str(x), info)))
            # self.earlystopping(score, self.model)
            self.save_model(epoch)
            # if self.earlystopping.early_stop:
            #     break

    def eval_bert(self, dataset, collate_fn, gen_out_path):
        self.model.eval()
        train_loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1, shuffle=False)
        out_arr = []
        with torch.no_grad():
            start_time = time()
            for step, batch_data in tqdm(enumerate(train_loader)):
                result = self.model.rank_generate(batch_data)
                out_arr.append(result)
            elapsed_time = time() - start_time
            print('elasped_time', elapsed_time)
            torch.save(out_arr, gen_out_path)

    def generate_seq(self, dataset, collate_fn, gen_out_path):      # only for test generation
        self.model.eval()
        train_loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=5, shuffle=False)
        gen_out_arr = []
        ref_arr = []
        with torch.no_grad():
            start_time = time()
            for step, batch_data in tqdm(enumerate(train_loader)):
                ref_arr.append(batch_data['decoder_word_tensor'][:, 1:])
                gen_out = self.model.generate(batch_data)
                gen_out_arr.append(gen_out)
            gen_out = torch.cat(gen_out_arr, dim=0).cpu().numpy()
            ref_arr = torch.cat(ref_arr, dim=0).cpu().numpy()
            elapsed_time = time() - start_time
            print('elasped_time', elapsed_time)
            need2save = {'gen_out': gen_out, 'gen_ref': ref_arr}
            torch.save(need2save, gen_out_path)

    def gen_case(self, train_arr: Dict, eval_epoch):
        pass












