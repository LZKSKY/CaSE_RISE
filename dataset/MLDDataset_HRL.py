from torch.utils.data import Dataset
from tqdm import tqdm
from config_n import device
from config_n import default_config as config
import torch
from copy import deepcopy
from dataset.utils import pad
from Preprocess.utils_ import Tag, obtain_gen_seq, TagSeq
from typing import List
import numpy as np
tag_seq = TagSeq()


class MLDDatasetHRL(Dataset):
    def __init__(self, samples=None, tokenizer=None, phrase_tokenizer=None, query_len=30, answer_len=30, n=1E10,
                 context_len=3, max_candidate_num=20, sample_tensor=None, data_type='gen'):
        super().__init__()
        self.samples = samples
        self.context_len = context_len
        self.max_candidate_num = max_candidate_num
        self.phrase_tokenizer = phrase_tokenizer
        self.max_gen_len = 20
        self.max_gen_num = 5
        self.tag_seq = tag_seq
        # if sample_tensor is None:
        self.query_len = query_len
        self.answer_len = answer_len
        self.extract_len = self.context_len * self.query_len + self.context_len * self.answer_len + 10    # 190
        self.final_len = self.extract_len + self.query_len  # 220
        self.gen_len = self.final_len + self.context_len + 1
        self.input_len = self.query_len + 2
        self.tokenizer = tokenizer
        self.edit2id = config.edit2id
        self.edit_pad_id = config.edit_pad_id
        self.n = n
        self.len = -1

        self.sample_tensor = []
        if data_type in ['gen', 'g']:
            self.load_edit_gen()
        elif data_type == 'eval':
            self.load_eval()
        else:
            raise ValueError
        # else:
        #     self.sample_tensor = sample_tensor
        #     self.len = len(self.sample_tensor)

    def obtain_gen_label(self, tag_arr: List[Tag], input_query):
        # classification
        gen_seq_arr = obtain_gen_seq(tag_arr, self.tokenizer.sep_token_id, input_query)
        seq_arr = []
        label_arr = []
        for seq in gen_seq_arr:
            decoder_seq = seq[0] + [self.tokenizer.cls_token_id]
            # decoder_seq.extend([self.tokenizer.bos_token_id] + seq[1])
            decoder_label = self.phrase_tokenizer.convert_token_to_id(seq[1])
            decoder_seq = pad(decoder_seq, self.tokenizer.pad_token_id, self.max_gen_len, padding_mode='l')
            seq_arr.append(decoder_seq)
            label_arr.append(decoder_label)
        seq_arr = pad(seq_arr, [self.tokenizer.pad_token_id] * self.max_gen_len, self.max_gen_num)
        label_arr = pad(label_arr, 0, self.max_gen_num)
        return seq_arr, label_arr

    def obtain_input_output(self, context_token, seq, seq_out):
        pad_token_id, sep_token_id, cls_token_id, bos_token_id, eos_token_id = \
            self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, \
            self.tokenizer.bos_token_id, self.tokenizer.eos_token_id

        in_seq = [bos_token_id] + seq[:self.query_len] + [eos_token_id]
        tag_label: List[Tag] = self.tag_seq.get_label(seq, seq_out, return_length=False)
        output_tag = [self.edit2id[edit_tag.ope] for edit_tag in tag_label + [Tag('K', [])]]
        decoder_input_seq, decoder_out_seq = self.obtain_gen_label(tag_arr=tag_label, input_query=in_seq)

        input_ids = context_token['input_ids'] + pad(in_seq[:self.query_len], pad_token_id, self.input_len)
        pos_ids = context_token['pos_ids'] + list(range(self.input_len))
        seg_ids = [0] * self.extract_len + [1] * self.input_len
        edit_pred_arr = [self.edit_pad_id] * self.extract_len + pad(output_tag, self.edit_pad_id, self.input_len)
        return input_ids, pos_ids, seg_ids, edit_pred_arr, decoder_input_seq, decoder_out_seq

    def obtain_input(self, context_token, seq):
        pad_token_id, sep_token_id, cls_token_id, bos_token_id, eos_token_id = \
            self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, \
            self.tokenizer.bos_token_id, self.tokenizer.eos_token_id

        in_seq = [bos_token_id] + seq[:self.query_len] + [eos_token_id]
        input_ids = context_token['input_ids'] + pad(in_seq[:self.query_len], pad_token_id, self.input_len)
        pos_ids = context_token['pos_ids'] + list(range(self.input_len))
        seg_ids = [0] * self.extract_len + [1] * self.input_len
        return input_ids, pos_ids, seg_ids

    def obtain_context(self, context, context_answer):
        assert len(context) == len(context_answer)
        context_token = {'input_ids': [self.tokenizer.cls_token_id], 'pos_ids': [0], 'type_ids': [0]}
        for index in range(len(context)):
            # [CLS] q [SEP] a [SEP] q [SEP] a
            seq1 = context[index][:self.query_len] + [self.tokenizer.sep_token_id]
            seq2 = context_answer[index][:self.query_len] + [self.tokenizer.sep_token_id]
            context_token['input_ids'] += seq1 + seq2
            if index == 0:
                context_token['pos_ids'] += list(range(1, len(seq1) + 1)) + list(range(len(seq2)))
            else:
                context_token['pos_ids'] += list(range(len(seq1))) + list(range(len(seq2)))
        context_token['input_ids'] = pad(context_token['input_ids'], self.tokenizer.pad_token_id, self.extract_len)
        context_token['pos_ids'] = pad(context_token['pos_ids'], 0, self.extract_len)
        return context_token

    def obtain_reward(self, sample):
        input_query, output_query = sample['input_query'], sample['output_query']
        trajectory = sample['trajectory']   # List[Seq]
        # if not trajectory:
        #     trajectory = [input_query, output_query]
        # trajectory.extend([output_query])
        Lev = [self.tag_seq.get_label(seq_t, output_query, return_length=True)[1] for seq_t in trajectory]
        Lev = np.array(Lev)
        reward = np.sqrt(Lev[:-1] - Lev[1:])    # sqrt(D_{t+1} - D_t)
        reward[-1] = max(np.sqrt(Lev[0]) / 2, 1)       # for final keep probability, mean of sqrt Lev distance
        for i in range(len(reward) - 2, -1, -1):
            reward[i] = reward[i + 1] * 0.5 + reward[i]
        # we assume trajectory has max length 4, then, we require, each time it needs to reduce sqrt(Lev) / 4, right?
        reward -= np.sqrt(Lev[0]) / 4
        return reward

    def load_edit_gen_V1(self):     # edit can do with gen
        pad_token_id, sep_token_id, cls_token_id, bos_token_id, eos_token_id = \
            self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, \
            self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        for sample in tqdm(self.samples):
            context_token = self.obtain_context(sample['context'][-self.context_len:],
                                                sample['context_answer'][-self.context_len:])
            input_query, output_query = sample['input_query'], sample['output_query']
            pad_output_query = pad(output_query[:self.query_len], pad_token_id, self.query_len)
            input_ids, pos_ids, seg_ids, edit_pred_arr, decoder_input_seq, decoder_out_seq = \
                self.obtain_input_output(context_token, input_query, output_query)
            self.sample_tensor.append([context_token['input_ids'], context_token['pos_ids'], context_token['type_ids'],
                                       edit_pred_arr, decoder_input_seq, decoder_out_seq, pad_output_query])
        self.len = len(self.sample_tensor)

    def load_edit_gen(self):     # edit can do with gen
        self.sample_tensor = []
        for sample_index, sample in enumerate(tqdm(self.samples)):
            context_token = self.obtain_context(sample['context'][-self.context_len:],
                                                sample['context_answer'][-self.context_len:])
            input_query, output_query = sample['input_query'], sample['output_query']
            if sample.get('trajectory', None) is None:
                sample['trajectory'] = [input_query]
                sample['reward'] = []
            trajectory = sample['trajectory']
            reward_arr = self.obtain_reward(sample)
            sample_arr = []
            for i in range(len(trajectory)):
                input_ids, pos_ids, seg_ids, edit_pred_arr, decoder_input_seq, decoder_out_seq = \
                    self.obtain_input_output(context_token, trajectory[i], output_query)
                sample_arr.append([input_ids, pos_ids, seg_ids, edit_pred_arr, decoder_input_seq,
                                   decoder_out_seq, reward_arr[i], sample_index])
            self.sample_tensor.extend(sample_arr)
        self.len = len(self.sample_tensor)

    def load_rl(self):     # edit can do with gen
        self.sample_tensor = []
        for sample_index, sample in enumerate(tqdm(self.samples)):
            context_token = self.obtain_context(sample['context'][-self.context_len:],
                                                sample['context_answer'][-self.context_len:])
            input_query, output_query = sample['input_query'], sample['output_query']
            if sample.get('trajectory', None) is None:
                sample['trajectory'] = [input_query]
                sample['reward'] = []
            trajectory = sample['trajectory']
            reward_arr = self.obtain_reward(sample)
            sample_arr = []
            for i in range(len(trajectory)):
                input_ids, pos_ids, seg_ids, edit_pred_arr, decoder_input_seq, decoder_out_seq = \
                    self.obtain_input_output(context_token, trajectory[i], output_query)
                sample_arr.append([input_ids, pos_ids, seg_ids, edit_pred_arr, decoder_input_seq,
                                   decoder_out_seq, reward_arr[i], sample_index])
            self.sample_tensor.extend(sample_arr)
        self.len = len(self.sample_tensor)

    def load_eval(self, new_sample_ids=None):
        for index, sample in enumerate(tqdm(self.samples)):
            context_token = self.obtain_context(sample['context'][-self.context_len:],
                                                sample['context_answer'][-self.context_len:])
            input_query, output_query = sample['input_query'], sample['output_query']
            # pad_output_query = pad(output_query[:self.query_len], pad_token_id, self.query_len)
            if sample.get('eval_trajectory', None) is None:
                sample['eval_trajectory'] = [input_query]
            if new_sample_ids is not None:
                sample['eval_trajectory'].extend(new_sample_ids[index])
            seq_input = sample['eval_trajectory'][-1]
            input_ids, pos_ids, seg_ids = self.obtain_input(context_token, seq_input)
            self.sample_tensor.extend([input_ids, pos_ids, seg_ids])
        self.len = len(self.sample_tensor)

    def eval_reward(self):
        r_arr = []
        for index, sample in enumerate(tqdm(self.samples)):
            input_query, output_query = sample['input_query'], sample['output_query']
            trajectory = sample['eval_trajectory']  # List[Seq]
            Lev = [self.tag_seq.get_label(seq_t, output_query, return_length=True)[1] for seq_t in trajectory]
            Lev = np.array(Lev)
            r = Lev[0] - Lev[-1]
            r_arr.append(r)
        return np.mean(r_arr)

    def obtain_data2delete(self):
        index = 0
        self.sample_tensor = []
        for sample in tqdm(self.samples):
            context_token = self.obtain_context(sample['context'][-self.context_len:],
                                                sample['context_answer'][-self.context_len:])
            # input_query, output_query = sample['input_query'], sample['output_query']
            trajectory = sample['trajectory']
            reward_arr = self.obtain_reward(sample)
            # sample_arr = []
            # len_traj = len(trajectory)
            # record which trans has been calculate and store at where
            record_map_arr = []
            for i in range(len(trajectory) - 1):
                input_ids, pos_ids, seg_ids, edit_pred_arr, decoder_input_seq, decoder_out_seq = \
                    self.obtain_input_output(context_token, trajectory[i], trajectory[i + 1])
                self.sample_tensor.append([input_ids, pos_ids, seg_ids, edit_pred_arr, decoder_input_seq,
                                           decoder_out_seq, reward_arr[i]])
                record_map_arr.append([i, i + 1, index])
                index += 1
            for i in range(len(trajectory) - 2):
                input_ids, pos_ids, seg_ids, edit_pred_arr, decoder_input_seq, decoder_out_seq = \
                    self.obtain_input_output(context_token, trajectory[i], trajectory[i + 2])
                self.sample_tensor.append([input_ids, pos_ids, seg_ids, edit_pred_arr, decoder_input_seq,
                                           decoder_out_seq, reward_arr[i]])
                record_map_arr.append([i, i + 1, index])
                index += 1
            sample['map_index_record'] = record_map_arr
        self.len = len(self.sample_tensor)

    def apply_delete(self, scores):
        for sample in tqdm(self.samples):
            trajectory = sample['trajectory']
            len_traj = len(trajectory)
            score_matrix = np.zeros([len_traj, len_traj])
            for tup in sample['map_index_record']:
                score_matrix[tup[0], tup[1]] = scores[tup[2]]
            new_trajectory = [trajectory[0]]
            fix = False
            for i in range(1, len_traj - 1):
                if not fix and score_matrix[i - 1, i] + score_matrix[i, i + 1] < score_matrix[i - 1, i + 1]:
                    fix = True
                else:
                    new_trajectory.append(trajectory[i])
                    fix = False
            new_trajectory.append(trajectory[-1])
            sample['trajectory'] = new_trajectory

    def __getitem__(self, index):
        return self.sample_tensor[index]

    def __len__(self):
        return self.len


def train_edit_gen_fn(batch_data):
    tokens, pos_ids, type_tokens, edit_labels, decoder_input, decoder_out, reward, sample_index = list(zip(*batch_data))
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
    pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long).to(device)
    type_tokens_tensor = torch.tensor(type_tokens, dtype=torch.long).to(device)
    edit_labels_tensor = torch.tensor(edit_labels, dtype=torch.long).to(device)
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.long).to(device)
    decoder_out_tensor = torch.tensor(decoder_out, dtype=torch.long).to(device)
    reward_tensor = torch.tensor(reward, dtype=torch.long).to(device)

    return {'input_token_tensor': tokens_tensor, 'pos_ids_tensor': pos_ids_tensor,
            'type_tokens_tensor': type_tokens_tensor, 'edit_labels_tensor': edit_labels_tensor,
            'decoder_input_tensor': decoder_input_tensor, 'decoder_out_tensor': decoder_out_tensor,
            'reward': reward_tensor, 'sample_index': sample_index}


def eval_fn(batch_data):
    tokens, pos_ids, type_tokens, edit_labels, decoder_input, decoder_out, reward = list(zip(*batch_data))
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
    pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long).to(device)
    type_tokens_tensor = torch.tensor(type_tokens, dtype=torch.long).to(device)
    return {'input_token_tensor': tokens_tensor, 'pos_ids_tensor': pos_ids_tensor,
            'type_tokens_tensor': type_tokens_tensor}














