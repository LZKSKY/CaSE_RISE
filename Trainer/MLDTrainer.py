import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from time import time
from Trainer.common import EarlyStopping
from config_n import Config
import numpy as np
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from Model.BertMLD import BertMLD
from dataset.MLDDataset_RL import MLDDatasetRL
config = Config()
logger = config.logger


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


def eval_metrics_loss(output, *args):
    key1 = args[0] if len(args) > 0 else 'gen_loss'
    out_arr = []
    for obj in output:
        out_arr.append(obj[key1])
    return -np.mean(out_arr)


def edit_gen_eval_metrics(output):
    effect_score_arr = []
    observe_score_arr = []
    gen_score = eval_metrics_loss(output)
    effect_score_arr.append(gen_score)
    if 'edit_loss' in output[0]:
        effect_score_arr.append(eval_metrics_loss(output, 'edit_loss'))
    if 'edit_out' in output[0]:
        observe_score_arr.append(edit_eval_metrics(output))
    return effect_score_arr, observe_score_arr


class MLDTrainer:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.batch_size = self.batch_size if hasattr(self, 'batch_size') else 32
        self.accumulation_steps = self.accumulation_steps if hasattr(self, 'accumulation_steps') else 1
        self.model_name = self.model_name if hasattr(self, 'model_name') else 'baseline'
        self.ema_rate = self.ema_rate if hasattr(self, 'ema_rate') else 0.995
        self.max_epoch = self.max_epoch if hasattr(self, 'max_epoch') else 25
        self.initial_lr = self.initial_lr if hasattr(self, 'initial_lr') else 5e-5
        self.tune_lr = self.tune_lr if hasattr(self, 'tune_lr') else 5e-5
        self.tune_epoch = self.tune_epoch if hasattr(self, 'tune_epoch') else 20
        self.train_size = self.train_size if hasattr(self, 'train_size') else 1e-3
        self.load_epoch = self.load_epoch if hasattr(self, 'load_epoch') else -1
        self.model: BertMLD = self.model if hasattr(self, 'model') else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.accumulation_count = 0
        self.begin_epoch = -1
        # self.optimizer = None
        self.optimizer = self.set_optimizer(self.initial_lr)
        model_bp_count = (self.max_epoch * self.train_size) / (self.batch_size * self.accumulation_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=min(int(model_bp_count / 10), 20000),
                                                         num_training_steps=int(model_bp_count) + 100)
        self.save_path = config.save_path + self.model_name
        self.rl_epoch = 10
        # self.earlystopping = EarlyStopping(patience=5, verbose=False, delta=0,
        #                                    path=self.save_path + self.model_name + '-checkpoint.pt',
        #                                    trace_func=logger.info)
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
            {'params': [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        logger.info('optimizer ')
        for tup in optimizer.param_groups:
            logger.info(f'optimizer lr = {tup["lr"]}, params len {len(tup["params"])}')
        return optimizer

    def train_batch(self, batch_data):
        self.accumulation_count += 1
        if isinstance(batch_data, (tuple, list)):
            batch_data, train_func = batch_data
        else:
            train_func = self.model.forward
        output, loss = train_func(batch_data, method=self.method)
        if isinstance(loss, tuple) or isinstance(loss, list):
            closs = [l.mean().cpu().item() for l in loss]
            loss = torch.cat([l.mean().reshape(1) for l in loss]).sum()
        else:
            loss = loss.sum()
            closs = [loss.cpu().item()]
        loss = loss / self.accumulation_steps
        loss.backward()

        if self.accumulation_count % self.accumulation_steps == 0:
            clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
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
            loss = self.train_batch((batch_data, self.model.forward))
            loss_arr.append(loss)
            if step > 0 and step % 10000 == 0:
                elapsed_time = time() - start_time
                loss = np.mean(np.stack(loss_arr, axis=0), axis=0)
                info = ['Method', self.model_name, 'Epoch', epoch, 'Batch ', step, 'Loss ',
                        loss, 'Time ', elapsed_time]
                if self.scheduler is not None:
                    info.extend(['Learning rate ', self.scheduler.get_last_lr()])
                logger.info(' '.join(map(lambda x: str(x), info)))
        loss = np.mean(np.stack(loss_arr, axis=0), axis=0)
        info = ['Method', self.model_name, 'Epoch', epoch, 'All ', 'Loss ', loss]
        logger.info(' '.join(map(lambda x: str(x), info)))
        return loss

    def eval_batch(self, batch_data):
        if isinstance(batch_data, (tuple, list)):
            batch_data, train_func = batch_data
        else:
            train_func = self.model.forward
        output = train_func(batch_data, method=self.method)
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
                metrics = self.eval_batch((batch_data, self.model.forward))
                metrics_arr.append(metrics)
        elapsed_time = time() - start_time
        effect_score, observe_score = edit_gen_eval_metrics(metrics_arr)
        info = ["Eval\t\t", 'Method', self.model_name, 'Epoch', epoch, 'Effect Score ', effect_score,
                'Observe Score ', observe_score, 'Time ', elapsed_time]
        logger.info([' '.join(map(lambda x: str(x), info))])
        return effect_score

    def gen_batch(self, batch_data):
        if isinstance(batch_data, (tuple, list)):
            batch_data, train_func = batch_data
        else:
            train_func = self.model.forward
        output = train_func(batch_data, method='eval')
        return output

    def gen_epoch(self, eval_dataset: MLDDatasetRL, eval_collate_fn, epoch=-1):
        self.model.eval()
        with torch.no_grad():
            eval_loader = DataLoader(eval_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size * 2,
                                     shuffle=True, pin_memory=False)
            start_time = time()
            seqs_arr = []
            for step, batch_data in tqdm(enumerate(eval_loader)):
                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor):
                        batch_data[key] = batch_data[key].to(self.device)
                seq = self.gen_batch((batch_data, self.model.generate_edit_gen))
                seqs_arr.extend(seq)
        elapsed_time = time() - start_time
        print(f'elapsed time: {elapsed_time}')
        return seqs_arr

    def save_model(self, epoch):
        save_name = self.save_path + f'-{epoch}'
        logger.info(f"model saved in {save_name}")
        torch.save(self.model.state_dict(), save_name)

    def load_model(self, epoch):
        if epoch < 0:
            return
        save_name = self.save_path + f'-{epoch}'
        if os.path.exists(save_name):
            self.begin_epoch = epoch
            logger.info(f"model load in {save_name}")
            self.model.load_state_dict(torch.load(save_name, map_location=self.device))
        else:
            logger.info(f"Not find {save_name}")

    def train_mld(self, train_dataset, train_collate_fn, eval_dataset, eval_collate_fn):
        logger.info(f"model_name {self.model_name} begin epoch {self.begin_epoch + 1}, max epoch {self.max_epoch}")
        for epoch in range(self.begin_epoch + 1, self.max_epoch):
            if epoch == self.tune_epoch:
                for para in self.model.parameters():
                    para.requires_grad = True
                self.set_optimizer(self.tune_lr)
            self.method = 'train'
            loss = self.train_epoch(train_dataset, train_collate_fn, epoch)
            self.method = 'eval'
            score = self.eval_epoch(eval_dataset, eval_collate_fn, epoch)
            # if epoch >= self.tune_epoch:
            #     self.earlystopping(sum(score), self.model)
            self.save_model(epoch)

    def generate_mld(self, eval_dataset, eval_collate_fn, gen_path):
        logger.info(f"generating sequence in epoch {self.load_epoch}")
        self.load_model(self.load_epoch)
        gen_seqs = self.gen_epoch(eval_dataset, eval_collate_fn)
        torch.save(gen_seqs, gen_path)

    def eval_rl(self, eval_dataset: MLDDatasetRL, eval_collate_fn, epoch=-1):      # by R[s_0]
        self.model.eval()
        with torch.no_grad():
            for i in range(4):
                eval_loader = DataLoader(eval_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size * 2,
                                         shuffle=True, pin_memory=False)
                start_time = time()
                gen_seqs_arr = []
                for step, batch_data in tqdm(enumerate(eval_loader)):
                    for key, value in batch_data.items():
                        if isinstance(value, torch.Tensor):
                            batch_data[key] = batch_data[key].to(self.device)
                    gen_seqs = self.model.generate_edit_gen(batch_data, method=None)
                    gen_seqs_arr.extend(gen_seqs)
                elapsed_time = time() - start_time
                eval_dataset.load_eval(gen_seqs_arr)
        reward_score = eval_dataset.eval_reward()
        info = ["Eval\t\t", 'Method', self.model_name, 'Epoch', epoch, 'Effect Score ', reward_score, 'Time ', elapsed_time]
        logger.info([' '.join(map(lambda x: str(x), info))])
        return reward_score

    def prob_epoch(self, eval_dataset, eval_collate_fn, epoch=-1):
        self.model.eval()
        with torch.no_grad():
            eval_loader = DataLoader(eval_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size * 2,
                                     shuffle=False, pin_memory=False)
            start_time = time()
            result_arr = []
            for step, batch_data in tqdm(enumerate(eval_loader)):
                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor):
                        batch_data[key] = batch_data[key].to(self.device)
                result = self.eval_batch((batch_data, self.model.forward))
                result_arr.extend(result)
        elapsed_time = time() - start_time

        info = ["Eval\t\t", 'Method', self.model_name, 'ProbEpoch', epoch, 'Time ', elapsed_time]
        logger.info([' '.join(map(lambda x: str(x), info))])
        return result_arr

    def extend_path(self, train_dataset: MLDDatasetRL, train_collate_fn, epoch=-1):
        # first delete and then insert
        self.model.eval()
        train_dataset.obtain_data2delete()
        self.method = 'train'
        result = self.prob_epoch(train_dataset, train_collate_fn, epoch=epoch)

    # TODO； implement train, eval, generate_sample, save_mid_data
    def train_mle(self, train_dataset, train_collate_fn, eval_dataset, eval_collate_fn):
        self.method = 'rl_train'
        self.eval_rl(eval_dataset, eval_collate_fn, -1)
        for e in range(self.rl_epoch):
            self.method = 'rl_train'
            self.train_epoch(train_dataset, train_collate_fn, e)
            eval_score = self.eval_rl(eval_dataset, eval_collate_fn, e)
            self.save_model(f'{e}-{eval_score}')
            # TODO: check probability and reward, and decide whether to expand
            self.extend_path(train_dataset, train_collate_fn, e)
            torch.save(train_dataset, config.quac_dataset_path + f'bert_iter_{e + 1}.train.pkl')
















