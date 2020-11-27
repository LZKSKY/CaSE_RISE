import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from time import time
import numpy as np
from dataset.MLDDataset_HRL import MLDDatasetHRL, tag_seq, pad
from dataset.MLDDataset_HRL import train_edit_gen_fn as collate
from dataset.DummyDataset import DummyDataset
from Trainer.MLDTrainer import MLDTrainer, config, logger
from Trainer.utils import cal_cost, sample_action_by_p, apply_action
from config_n import default_config as config
from copy import deepcopy


class RLTrainer(MLDTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.L = config.L
        self.dummy_dataset = DummyDataset([])

    def get_seq2prob(self, action_arr, input_query, dataset: MLDDatasetHRL, sample_index):
        sample_arr = []
        for tag_arr in action_arr:
            decoder_input_seq, decoder_out_seq = dataset.obtain_gen_label(tag_arr, input_query)
            edit_pred_arr = [dataset.edit_pad_id] * dataset.extract_len + \
                pad(tag_arr, dataset.edit_pad_id, dataset.input_len)
            sample_tensor = deepcopy(dataset.sample_tensor[sample_index])
            sample_tensor[3:5] = edit_pred_arr, decoder_input_seq, decoder_out_seq
            sample_arr.append(sample_tensor)
        self.dummy_dataset.samples = sample_arr

    def prob_dataset(self):
        train_loader = DataLoader(self.dummy_dataset, collate_fn=collate, batch_size=len(self.dummy_dataset),
                                  shuffle=False)
        for batch_data in train_loader:
            output, loss = self.model.forward(batch_data, method='prob')
            prob: torch.Tensor = output['edit_loss'] + output['gen_loss']
            return prob.detach().cpu().numpy()

    @staticmethod
    def get_Lev(x, y):
        edit_matrix = tag_seq.get_edit_matrix(x, y)
        return edit_matrix[-1][-1]

    @staticmethod
    def get_reward(y_t, y_t1, y):
        d0 = RLTrainer.get_Lev(y_t, y)
        d1 = RLTrainer.get_Lev(y_t1, y)
        r1 = np.sqrt(np.abs(d0 - d1)) * (d0 >= d1)      # gain on minimizing objective
        r2 = 1 / (d1 + 1)                        # gain of current state
        return r1 + r2

    def cal_cost_main(self, batch_data, dataset: MLDDatasetHRL):
        self.model.eval()
        gen_dict = self.model.do_edit_gen(batch_data, compute_loss=False)
        assert 'gen_out' in gen_dict
        gen_prob = gen_dict['gen_out'][:, 190:].softmax(dim=-1).detach().cpu().numpy()       # [b, s, 4 + 1]
        sample_index_arr = batch_data['sample_index']
        for index, sample_index in enumerate(sample_index_arr):
            sample = dataset.samples[sample_index]
            input_query, output_query = sample['input_query'], sample['output_query']
            prob = gen_prob[index]
            C, P = cal_cost(seq_a=input_query, seq_b=output_query, prob=prob)
            action_arr = []
            for l in range(self.L):
                tag_arr = sample_action_by_p(seq_a=input_query, seq_b=output_query, P=P)
                action_arr.append(tag_arr)
            self.get_seq2prob(action_arr, input_query, dataset, sample_index)
            prob = self.prob_dataset()
            sel_action_index = prob.argmax()
            sel_action = action_arr[sel_action_index]
            extend_input_query = [self.model.tokenizer.bos_token_id] + input_query
            new_query = apply_action(extend_input_query, sel_action)    # [BOS] + NEW_X
            reward = RLTrainer.get_reward(input_query, new_query[1:], output_query)




    def train_batch_rl(self, batch_data, dataset):
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

    def train_epoch_rl(self, train_dataset, train_collate_fn, epoch):
        self.model.train()
        train_loader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size,
                                  shuffle=True, pin_memory=False)
        loss_arr = []
        start_time = time()
        for step, batch_data in tqdm(enumerate(train_loader)):
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    batch_data[key] = batch_data[key].to(self.device)
            loss = self.train_batch_rl((batch_data, self.model.forward))
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

    # TODO； implement train, eval, generate_sample, save_mid_data
    def train_rl(self, train_dataset, train_collate_fn, eval_dataset, eval_collate_fn):
        self.method = 'rl_train'
        self.eval_rl(eval_dataset, eval_collate_fn, -1)
        for e in range(self.rl_epoch):
            self.method = 'rl_train'
            self.train_epoch_rl(train_dataset, train_collate_fn, e)
            eval_score = self.eval_rl(eval_dataset, eval_collate_fn, e)
            self.save_model(f'{e}-{eval_score}')
            # TODO: check probability and reward, and decide whether to expand
            self.extend_path(train_dataset, train_collate_fn, e)
            torch.save(train_dataset, config.quac_dataset_path + f'bert_iter_{e + 1}.train.pkl')



















