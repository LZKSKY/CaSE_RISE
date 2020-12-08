import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from time import time
import numpy as np
from dataset.MLDDataset_HRL import MLDDatasetHRL, pad
from dataset.MLDDataset_HRL import train_edit_gen_fn as collate
from dataset.MLDDataset_HRL import eval_gen_fn as eval_collate
from dataset.DummyDataset import DummyDataset
from Trainer.MLDTrainer import MLDTrainer, config, logger
from config_n import device
from Trainer.utils import cal_cost, sample_action_by_p, apply_action, K_sampling, sampling
# from config_n import Config
from copy import deepcopy
from tensorboardX import SummaryWriter


class RLTrainer(MLDTrainer):
    def __init__(self, train_dataset=None, train_collate=None, sampling_strategy='RISE', **kwargs):
        super().__init__(**kwargs)
        self.L = config.L
        self.train_sample = config
        self.train_fold = config.train_fold
        self.sample_fold = config.sample_fold
        self.rl_begin_epoch = config.rl_begin_epoch
        self.rl_epoch = config.rl_epoch
        self.load_epoch = config.load_epoch
        self.dummy_dataset = DummyDataset([])
        self.dataset: MLDDatasetHRL = train_dataset
        self.collate_fn = train_collate
        self.summary_writer = SummaryWriter(log_dir=config.log_path, filename_suffix=self.model_name)
        # self.sampling_strategy = 'softmax'
        self.sampling_strategy = sampling_strategy

    def get_action_prob(self):
        train_loader = DataLoader(self.dummy_dataset, collate_fn=collate, batch_size=self.batch_size * 4,
                                  shuffle=False)
        all_prob = []
        self.model.eval()
        with torch.no_grad():
            for batch_data in train_loader:
                output = self.model.forward(batch_data, method=self.method)
                prob: torch.Tensor = output['edit_loss'] + output['gen_loss']
                all_prob.append(-prob)  # negative loss
            all_prob = torch.cat(all_prob, dim=0)   # [b]
            all_prob = all_prob.contiguous().view(-1, self.L)
            return all_prob.detach().cpu().numpy()

    def get_edit_prob(self, train_loader):
        self.model.eval()
        info_arr = {'edit_prob': [], 'sample_index': []}
        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(train_loader, desc='get_edit_prob')):
                gen_dict = self.model.edit_pred(batch_data, compute_loss=False, method=self.method)
                assert 'edit_output' in gen_dict
                edit_prob = gen_dict['edit_output'][:, 190:].softmax(dim=-1).detach().cpu().numpy()       # [b, s, 4 + 1]
                edit_prob[:, :, 0] = 0.
                sample_index_arr = batch_data['sample_index']
                info_arr['edit_prob'].extend(edit_prob)
                info_arr['sample_index'].extend(sample_index_arr)
            return info_arr

    def get_phrase_prob(self):
        train_loader = DataLoader(self.dummy_dataset, collate_fn=eval_collate, batch_size=self.batch_size * 4,
                                  shuffle=False)
        info_arr = {'gen_prob': [], 'sample_index': []}
        self.model.eval()
        with torch.no_grad():
            for batch_data in train_loader:
                output = self.model.forward(batch_data, method=self.method, compute_loss=False)
                gen_prob: torch.Tensor = output['gen_output'].softmax(dim=-1).detach().cpu().numpy()
                gen_prob[:, :, 0] = 0.
                sample_index_arr = batch_data['sample_index']
                info_arr['gen_prob'].extend(gen_prob)
                info_arr['sample_index'].extend(sample_index_arr)
            return info_arr

    @staticmethod
    def obtain_cache(action_prob, wait_arr):
        sel_action_arr = action_prob.argmax(axis=1)
        assert len(sel_action_arr) == len(wait_arr)
        add_cache_arr = []
        for index, wait in enumerate(wait_arr):
            cand_action_arr, extend_input_query, sample_index = wait
            sel_action_index = sel_action_arr[index]
            sel_action = cand_action_arr[sel_action_index]      # m + 2
            new_query = apply_action(extend_input_query, sel_action)    # [BOS] + NEW_X + [EOS]
            add_cache_arr.append({'edits': sel_action, 'new_query': new_query[1:-1], 'sample_index': sample_index})
        return add_cache_arr

    def obtain_cache_greedy(self, gen_prob, wait_arr):
        # gen_prob = gen_prob.argmax(axis=1)
        assert len(gen_prob['gen_prob']) == len(wait_arr)
        add_cache_arr = []
        for index, wait in enumerate(wait_arr):
            edits, extend_input_query, sample_index = wait
            prob = gen_prob['gen_prob'][index]
            sample_id = sampling(prob, L=len(prob[0]), sampling_strategy=self.sampling_strategy, tag=False)  # [b, s]
            self.dataset.obtain_edits(edits, sample_id)
            new_query = apply_action(extend_input_query, edits)    # m + 2
            add_cache_arr.append({'edits': edits, 'new_query': new_query[1:-1], 'sample_index': sample_index})
        return add_cache_arr

    def sample_action(self, dataset, info_arr):
        wait_arr = []
        edit_prob = info_arr['edit_prob']
        sample_index_arr = info_arr['sample_index']
        for index, sample_index in enumerate(tqdm(sample_index_arr, desc='sample_action')):
            sample: dict = deepcopy(dataset.sample_used['sample'][sample_index])
            input_query, output_query = sample['current_output_query'], sample['output_query']
            prob = edit_prob[index]
            C, P = cal_cost(seq_a=input_query, seq_b=output_query, prob=prob)
            action_arr = []
            for l in range(self.L):
                tag_arr = sample_action_by_p(seq_a=input_query, seq_b=output_query, P=P)    # [m + 2]
                K_sampling(tag_arr, prob)   # [m + 2]
                action_arr.append(tag_arr)
            # tag_arr = sample_action_by_p(seq_a=input_query, seq_b=output_query, P=P)  # [m + 2]
            # K_sampling(tag_arr, prob)  # [m + 2]
            # action = tag_arr
            extend_input_query = [self.model.tokenizer.bos_token_id] + input_query + [self.model.tokenizer.eos_token_id]
            # self.get_seq2prob(action_arr, extend_input_query, dataset, sample_index)
            wait_arr.append([action_arr, extend_input_query, sample_index])
            # wait_arr.append([action, extend_input_query, sample_index])
        return wait_arr

    def sample_step(self, train_dataset, train_collate_fn, train_loader=None, max_count=10, verbose=False, fold=-1):
        self.method = 'prob'
        # =====================================
        # max_count_clip = max_count / 4
        train_dataset.sample_tensor_used = train_dataset.sample_used['sample_tensor']
        # =====================================
        if train_loader is None:
            logger.debug(f'new dataset loader in sample_step with dataset length {len(train_dataset)}')
            train_loader = DataLoader(train_dataset, collate_fn=eval_collate, batch_size=self.batch_size * 4,
                                      shuffle=False, pin_memory=False)
        start_time = time()
        self.dummy_dataset.samples = []
        t1 = time()
        info_arr = self.get_edit_prob(train_loader)
        t2 = time()
        wait_arr = self.sample_action(train_dataset, info_arr)
        t3 = time()
        self.dummy_dataset.samples = train_dataset.load_sample_prob(wait_arr)
        t4 = time()
        action_prob = self.get_action_prob()
        del train_loader
        t5 = time()
        add_cache_arr = self.obtain_cache(action_prob, wait_arr)
        self.dataset.add_cache(add_cache_arr)
        t6 = time()
        t_arr = np.array([t1, t2, t3, t4, t5, t6])
        print(t_arr[1:] - t_arr[:-1])
        elapsed_time = time() - start_time
        print(f'sampled in fold {fold}')
        # =====================================
        train_dataset.sample_tensor_used = train_dataset.sample_used['tensor']
        # =====================================
        if verbose:
            info = ['Method', self.model_name, 'Sample: ', 'Time ', elapsed_time]
            if self.scheduler is not None:
                info.extend(['Learning rate ', self.scheduler.get_last_lr()])
            logger.info(' '.join(map(lambda x: str(x), info)))

    def sample_greedy(self, dataset, info_arr):
        self.model.eval()
        edit_prob = info_arr['edit_prob']
        sample_index_arr = info_arr['sample_index']
        wait_arr = []
        for index, sample_index in enumerate(sample_index_arr):
            sample: dict = deepcopy(dataset.sample_used['sample'][sample_index])
            input_query, output_query = sample['current_output_query'], sample['output_query']
            prob = edit_prob[index]     # [s, E]
            edits = sampling(prob, len(input_query) + 2, sampling_strategy='softmax', tag=True)     # m + 2
            extend_input_query = [self.model.tokenizer.bos_token_id] + input_query + [self.model.tokenizer.eos_token_id]
            wait_arr.append([edits, extend_input_query, sample_index])
        return wait_arr

    def sample_step_greedy(self, train_dataset, train_collate_fn, train_loader=None, max_count=10, verbose=False,
                           fold=-1):
        self.method = 'prob'
        # =====================================
        # max_count_clip = max_count / 4
        train_dataset.sample_tensor_used = train_dataset.sample_used['sample_tensor'][:max_count]
        # =====================================
        if train_loader is None:
            logger.debug(f'new dataset loader in sample_step')
            train_loader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=False)
        start_time = time()
        self.dummy_dataset.samples = []
        all_wait_arr = []
        info_arr = self.get_edit_prob(train_loader)
        wait_arr = self.sample_greedy(train_dataset, info_arr)
        self.dummy_dataset.samples = train_dataset.load_sample_prob_action(wait_arr)
        gen_out_prob = self.get_phrase_prob()
        del train_loader
        add_cache_arr = self.obtain_cache_greedy(gen_out_prob, all_wait_arr)
        self.dataset.add_cache(add_cache_arr)
        elapsed_time = time() - start_time
        print(f'sampled in fold {fold}')
        # =====================================
        train_dataset.sample_tensor_used = train_dataset.sample_used['tensor']
        # =====================================
        if verbose:
            info = ['Method', self.model_name, 'Sample: ', 'Time ', elapsed_time]
            if self.scheduler is not None:
                info.extend(['Learning rate ', self.scheduler.get_last_lr()])
            logger.info(' '.join(map(lambda x: str(x), info)))

    def update_para(self):
        clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()
        self.accumulation_count = 0

    def train_batch(self, batch_data):
        self.accumulation_count += 1
        out_dict = self.model.forward(batch_data, method=self.method, compute_loss=True)
        loss = [out_dict['gen_loss'], out_dict['edit_loss']]
        closs = [l.mean().cpu().item() for l in (loss + [out_dict['reward'].float()])]
        loss = torch.cat([l.mean().reshape(1) for l in loss]).sum()
        loss = loss / self.accumulation_steps
        loss.backward()
        if self.accumulation_count % self.accumulation_steps == 0:
            self.update_para()
        return closs

    def train_step(self, train_dataset, train_collate_fn, train_loader=None, max_count=10, verbose=False, fold=-1):
        self.model.train()
        self.method = 'rl_train'
        if train_loader is None:
            logger.debug(f'new dataset loader in train_step, with dataset length {len(train_dataset)}')
            # train_dataset.sample_tensor_used = train_dataset.sample_tensor_used[:max_count]
            train_loader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=False)
        loss_arr = []
        start_time = time()
        # trainer_loader = tqdm(train_loader, desc=f'Fold {fold}-training')
        for step, batch_data in enumerate(train_loader):
            loss = self.train_batch(batch_data)
            loss_arr.append(loss)
        self.update_para()
        del train_loader
        loss = np.mean(np.stack(loss_arr, axis=0), axis=0)
        elapsed_time = time() - start_time
        print(f'trained in fold {fold}')
        if verbose:
            info = ['Method', self.model_name, 'Train: ', 'Loss ', loss, 'Time ', elapsed_time]
            if self.scheduler is not None:
                info.extend(['Learning rate ', self.scheduler.get_last_lr()])
            logger.info(' '.join(map(lambda x: str(x), info)))
        return loss_arr, loss

    def save_dataset(self, epoch):
        torch.save(self.dataset.samples, config.quac_dataset_path + f'bert_iter_{epoch}.{self.sampling_strategy}.pkl')

    def train_rl(self):
        epoch_batches = np.ceil(float(len(self.dataset.samples['origin'])) / self.batch_size)       # all batch nums in one dataset
        epoch_batches = int(epoch_batches)       # all batch nums in one dataset
        fold_train_batches = np.ceil(epoch_batches / self.train_fold)     # batch nums to train one fold
        fold_sample_batches = np.ceil(epoch_batches / self.sample_fold)     # batch nums to train one fold
        fold_train_batches = int(fold_train_batches)
        fold_sample_batches = int(fold_sample_batches)
        fold_all = int(self.train_fold * self.rl_epoch)        # all batch nums need to train
        begin_sample_fold = int(self.train_fold * self.rl_begin_epoch)
        load_epoch_fold = int(self.train_fold * self.load_epoch)
        self.load_model(self.load_epoch)
        self.dataset.sample_tensor_used_len = int(fold_train_batches * self.batch_size)
        self.dataset.load_sample_check()
        logger.info(f'All batch num: {epoch_batches}\n'
                    f'Train folds batches: {fold_train_batches}\n'
                    f'Sample folds batches: {fold_sample_batches}\n'
                    f'All train folds: {fold_all}\n'
                    f'Sample Epoch begin from: {begin_sample_fold}\n'
                    f'Load Epoch from fold: {load_epoch_fold}\n')
        epoch_loss = []
        if self.tune_epoch <= 0 or self.load_epoch >= self.tune_epoch:
            for para in self.model.parameters():
                para.requires_grad = True
            self.set_optimizer(self.tune_lr)
        for t in range(load_epoch_fold + 1, fold_all + 1):
            verbose = t % self.train_fold == 0
            loss_arr, loss = self.train_step(self.dataset, self.collate_fn, max_count=fold_train_batches * self.batch_size,
                                             verbose=False, fold=t)
            epoch_loss.extend(loss_arr)
            if t > begin_sample_fold:
                self.sample_step(self.dataset, self.collate_fn, max_count=fold_sample_batches * self.batch_size, verbose=False,
                                 fold=t)
                self.dataset.update_cache()
            if verbose:
                epoch = int(t / self.train_fold)
                loss = np.mean(np.stack(epoch_loss, axis=0), axis=0)
                epoch_loss = []
                self.summary_writer.add_scalar(tag='loss1', scalar_value=loss[0])
                self.summary_writer.add_scalar(tag='loss2', scalar_value=loss[1])
                # self.save_model(f'{self.model_name}-{epoch}.pkl')
                self.save_model(epoch)
                if t > begin_sample_fold:
                    self.save_dataset(epoch)
                if epoch == self.tune_epoch:
                    for para in self.model.parameters():
                        para.requires_grad = True
                    self.set_optimizer(self.tune_lr)
        self.summary_writer.close()

    def gen_seq4next_gen(self, eval_dataset: MLDDatasetHRL, batch_data, gen_seq_arr):
        for index, gen_seq in enumerate(gen_seq_arr):
            sample_index = batch_data['sample_index'][index]
            sample: dict = eval_dataset.samples['samples'][sample_index]
            sample['input_query'] = gen_seq['gen_query_ids']

    def gen_epoch(self, eval_dataset: MLDDatasetHRL, eval_collate_fn, epoch=-1):
        self.model.eval()
        with torch.no_grad():
            final_seq = [{'input_query': self.model.ids2str(sample['input_query']),
                          'output_query': self.model.ids2str(sample['output_query']),
                          'gen_query': [],
                          'edit_arr': []}
                         for sample in eval_dataset.samples['origin']]
            for gen_turn in range(1, 1 + self.L):
                eval_loader = DataLoader(eval_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size * 2,
                                         shuffle=False, pin_memory=False)
                start_time = time()
                for step, batch_data in tqdm(enumerate(eval_loader)):
                    gen_seq = self.model.generate_edit_gen(batch_data, method='eval')
                    self.gen_seq4next_gen(eval_dataset, batch_data, gen_seq_arr=gen_seq)
                    sample_index = batch_data['sample_index']
                    for index, seq in enumerate(gen_seq):
                        final_seq[sample_index[index]]['edit_arr'].append(seq['edit_ids'])
                        final_seq[sample_index[index]]['gen_query'].append(self.model.ids2str(seq['gen_query_ids']))
            elapsed_time = time() - start_time
            print(f'elapsed time: {elapsed_time}')
            return final_seq

    def generate_mld(self, eval_dataset, eval_collate_fn, gen_path):
        # logger.info(f"generating sequence in epoch {self.load_epoch}")
        gen_seqs = self.gen_epoch(eval_dataset, eval_collate_fn)
        torch.save(gen_seqs, gen_path)
        logger.info(f'generated data saved in {gen_path}')














