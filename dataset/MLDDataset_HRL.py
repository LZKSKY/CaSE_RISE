from torch.utils.data import Dataset
from config_n import device
from config_n import default_config as config
import torch
from copy import deepcopy
from dataset.utils import pad
from Preprocess.utils_ import Tag, obtain_gen_seq, TagSeq
from Preprocess.PhraseTokenizer import PhraseTokenizer
from typing import List
from random import shuffle
import numpy as np
tag_seq = TagSeq()


class MLDDatasetHRL(Dataset):
    def __init__(self, samples=None, tokenizer=None, phrase_tokenizer=None, query_len=30, answer_len=30, n=1E10,
                 context_len=3, max_candidate_num=20, sample_tensor=None, data_type='gen'):
        super().__init__()
        self.samples: dict = {'origin': samples, 'memory': [], 'cache': [],
                              'origin_tensor': [], 'memory_tensor': [], 'cache_tensor': [],
                              'origin_sample_tensor': [], 'memory_sample_tensor': [], 'cache_sample_tensor': []
                              }    # origin sample
        self.context_len = context_len
        self.max_candidate_num = max_candidate_num
        self.phrase_tokenizer: PhraseTokenizer = phrase_tokenizer
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
        self.max_memory_len = len(samples)
        '''
            samples: tensor for samples
            memory, cache: tensor for memory and cache
            memory_sample, cache_sample: tensor for sampling memory and cache
        '''
        '''
            sample, tensor, sample_tensor when in true memory
        '''
        self.sample_used: dict = {'sample': [], 'tensor': [], 'sample_tensor': []}
        self.sample_tensor_used: list = []
        self.sample_tensor_used_len = len(samples)

    def obtain_gen_label(self, tag_arr: List[Tag], input_query):
        gen_seq_arr = obtain_gen_seq(tag_arr, input_query)
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

    def obtain_gen_sequence(self, tag_arr: List[Tag], input_query):
        gen_seq_arr = obtain_gen_seq(tag_arr, input_query)
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

    def obtain_input_output(self, context_token, seq, seq_out=None, edits=None):
        pad_token_id, sep_token_id, cls_token_id, bos_token_id, eos_token_id = \
            self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, \
            self.tokenizer.bos_token_id, self.tokenizer.eos_token_id

        in_seq = [bos_token_id] + seq[:self.query_len] + [eos_token_id]
        input_ids = context_token['input_ids'] + pad(in_seq, pad_token_id, self.input_len)
        pos_ids = context_token['pos_ids'] + list(range(self.input_len))
        seg_ids = [0] * self.extract_len + [1] * self.input_len
        if edits is None:
            tag_label: List[Tag] = self.tag_seq.get_label(seq, seq_out, return_length=False)
        else:
            tag_label = edits[:-1] if len(edits) == len(in_seq) else edits
            assert len(tag_label) == len(in_seq) - 1
        tag_label += [Tag('K', [])]
        output_tag = [self.edit2id[edit_tag.ope] for edit_tag in tag_label]

        decoder_input_seq, decoder_out_seq = self.obtain_gen_label(tag_arr=tag_label, input_query=in_seq)
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

    @staticmethod
    def get_Lev(x, y):
        edit_matrix = tag_seq.get_edit_matrix(x, y)
        return edit_matrix[-1][-1][0]

    @staticmethod
    def get_reward(y_t, y_t1, y, edit_arr, d0=None, d1=None, reward_assign=config.reward_assign):
        if d1 is None:
            d1 = MLDDatasetHRL.get_Lev(y_t1, y)
        if d0 is None:
            d0 = MLDDatasetHRL.get_Lev(y_t, y)
        edit_no_K = np.sum(np.array(edit_arr) > 1)
        r = 1 / (d1 + 1)
        r *= 1 + 0.5 * max(d0 - d1 - edit_no_K + 1, -3)
        return min(r, 1.)

    def load_samples(self, samples, d1=None, verbose=False, k_in='input_query', k_out='current_output_query',
                     has_edits=False):
        sample_tensor = []
        for index, sample in enumerate(samples):
            context_token = self.obtain_context(sample['context'][-self.context_len:],
                                                sample['context_answer'][-self.context_len:])
            input_query, output_query = sample[k_in], sample[k_out]
            edits = sample['edits'] if has_edits else None
            input_ids, pos_ids, seg_ids, edit_pred_arr, decoder_input_seq, decoder_out_seq = \
                self.obtain_input_output(context_token, input_query, output_query, edits)
            reward = MLDDatasetHRL.get_reward(input_query, output_query, output_query, edit_pred_arr, d1=d1)
            sample['reward'] = reward
            sample_arr = [input_ids, pos_ids, seg_ids, edit_pred_arr, decoder_input_seq,
                          decoder_out_seq, reward]
            sample_tensor.append(sample_arr)
        return sample_tensor

    def load_samples2gen(self, samples, verbose=False, k_in='input_query'):
        sample_tensor = []
        for sample_index, sample in enumerate(samples):
            context_token = self.obtain_context(sample['context'][-self.context_len:],
                                                sample['context_answer'][-self.context_len:])
            input_query = sample[k_in]
            input_ids, pos_ids, seg_ids = self.obtain_input(context_token, input_query)
            sample_arr = [input_ids, pos_ids, seg_ids]
            sample_tensor.append(sample_arr)
        return sample_tensor

    def obtain_edits(self, edits: List[Tag], gen_ids):
        # return final action, decoder_label
        input_len = len(edits)
        i = 0
        gen_index = 0
        while i < input_len:
            if edits[i].ope == 'I':
                edits[i].seq = self.phrase_tokenizer.convert_id_to_token_arr(gen_ids[gen_index])
                gen_index += 1
                i += 1
            elif edits[i].ope == 'S':
                token_arr = self.phrase_tokenizer.convert_id_to_token_arr(gen_ids[gen_index])
                gen_index += 1
                edits[i].seq = token_arr
                i += 1
                while i < input_len and edits[i].ope == 'S':
                    edits[i].seq = token_arr
                    i += 1
            else:
                i += 1
            if gen_index >= len(gen_ids):
                break

    def load_sample_check(self):
        # load sample when loaded
        for sample in self.samples['origin']:
            sample['current_output_query'] = deepcopy(sample['input_query'])
        self.samples['origin_tensor'] = self.load_samples(self.samples['origin'], d1=0., k_in='input_query',
                                                          k_out='output_query', has_edits=False)
        self.samples['origin_sample_tensor'] = self.load_samples2gen(self.samples['origin'], k_in='current_output_query')
        self.samples['memory_tensor'] = self.load_samples(self.samples['memory'], k_in='input_query',
                                                          k_out='current_output_query', has_edits=True)
        self.samples['memory_sample_tensor'] = self.load_samples2gen(self.samples['memory'], k_in='current_output_query')
        self.append_index(self.samples['origin_tensor'])
        self.append_index(self.samples['origin_sample_tensor'])
        self.append_index(self.samples['memory_tensor'])
        self.append_index(self.samples['memory_sample_tensor'])

        shuff_data = list(zip(self.samples['memory'], self.samples['memory_tensor'], self.samples['memory_sample_tensor']))
        shuff_data.extend(zip(self.samples['origin'], self.samples['origin_tensor'], self.samples['origin_sample_tensor']))
        shuffle(shuff_data)
        shuff_data = shuff_data[-self.sample_tensor_used_len:]
        self.sample_used['sample'], self.sample_used['tensor'], self.sample_used['sample_tensor'] = list(zip(*shuff_data))
        self.set_index(self.sample_used['tensor'])
        self.set_index(self.sample_used['sample_tensor'])
        self.sample_tensor_used = self.sample_used['tensor']

    def load_sample_gen(self):
        self.sample_used['sample'] = self.samples['origin']
        self.sample_used['sample_tensor'] = self.load_samples2gen(self.samples['origin'], k_in='input_query')
        self.append_index(self.sample_used['sample_tensor'])
        self.sample_tensor_used = self.sample_used['sample_tensor']

    def load_sample_prob(self, wait_arr):
        # for multiple action
        all_sample_arr = []
        for action_arr, extend_input_query, sample_index in wait_arr:
            for action in action_arr:
                sample = deepcopy(self.sample_used['sample'][sample_index])
                sample['edits'] = action
                sample['output_query'] = extend_input_query[1:-1]
                all_sample_arr.append(sample)
        samples = self.load_samples(all_sample_arr, d1=0., k_in='current_output_query', k_out='output_query',
                                    has_edits=True)
        self.append_index(samples)
        return samples

    def load_sample_prob_action(self, wait_arr):
        # for simple action
        all_sample_arr = []
        for action, extend_input_query, sample_index in wait_arr:
            sample = deepcopy(self.sample_used['sample'][sample_index])
            sample['edits'] = action
            sample['output_query'] = extend_input_query[1:-1]
            all_sample_arr.append(sample)
        samples = self.load_samples(all_sample_arr, d1=0., k_in='current_output_query', k_out='output_query',
                                    has_edits=True)
        self.append_index(samples)
        return samples

    def add_cache(self, add_cache_arr):
        sample_arr = []
        for index, cache in enumerate(add_cache_arr):
            sample_index = cache['sample_index']
            new_query = cache['new_query'][:self.query_len]
            edits = cache['edits']
            sample: dict = deepcopy(self.sample_used['sample'][sample_index])
            sample['input_query'], sample['current_output_query'] = sample['current_output_query'], new_query
            sample['edits'] = edits
            # dup_samples = deepcopy(sample)
            # dup_samples['current_output_query'] = sample['output_query']
            sample_arr.append(sample)
            # sample_arr.append(dup_samples)
        self.samples['cache'].extend(sample_arr)

    @staticmethod
    def append_index(sample_tensor):
        for index, sample in enumerate(sample_tensor):
            sample.append(index)

    @staticmethod
    def set_index(sample_tensor):
        # print('setting index')
        for index, sample in enumerate(sample_tensor):
            sample[-1] = index

    def update_cache(self):
        cache_tensor = self.load_samples(self.samples['cache'], verbose=False, k_in='input_query',
                                         k_out='current_output_query', has_edits=True)
        cache_sample_tensor = self.load_samples2gen(self.samples['cache'], verbose=False, k_in='current_output_query')
        self.samples['cache_tensor'] = cache_tensor
        self.samples['cache_sample_tensor'] = cache_sample_tensor
        self.append_index(self.samples['cache_tensor'])
        self.append_index(self.samples['cache_sample_tensor'])
        # t2 = time()
        # random replace
        shuff_data = list(zip(self.samples['memory'], self.samples['memory_tensor'], self.samples['memory_sample_tensor']))
        shuff_data.extend(zip(self.samples['cache'], self.samples['cache_tensor'], self.samples['cache_sample_tensor']))
        shuffle(shuff_data)
        shuff_data = shuff_data[-self.max_memory_len:]
        self.samples['memory'],  self.samples['memory_tensor'], self.samples['memory_sample_tensor'] = list(zip(*shuff_data))

        shuff_data.extend(zip(self.samples['origin'], self.samples['origin_tensor'], self.samples['origin_sample_tensor']))
        shuffle(shuff_data)
        shuff_data = shuff_data[-self.sample_tensor_used_len:]
        self.sample_used['sample'], self.sample_used['tensor'], self.sample_used['sample_tensor'] = list(zip(*shuff_data))
        # t3 = time()
        self.set_index(self.sample_used['tensor'])
        self.set_index(self.sample_used['sample_tensor'])
        self.samples['cache'] = []
        self.samples['cache_tensor'] = []
        self.samples['cache_sample_tensor'] = []
        self.sample_tensor_used = self.sample_used['tensor']

    def update_cache_greedy(self):
        # t1 = time()
        cache_tensor = self.load_samples(self.samples['cache'], verbose=False, k_in='input_query',
                                         k_out='current_output_query', has_edits=True)
        cache_sample_tensor = self.load_samples2gen(self.samples['cache'], verbose=False, k_in='current_output_query')
        self.samples['cache_tensor'] = cache_tensor
        self.samples['cache_sample_tensor'] = cache_sample_tensor
        self.append_index(self.samples['cache_tensor'])
        self.append_index(self.samples['cache_sample_tensor'])
        # t2 = time()
        # random replace
        shuff_data = list(zip(self.samples['memory'], self.samples['memory_tensor'], self.samples['memory_sample_tensor']))
        shuff_data.extend(zip(self.samples['cache'], self.samples['cache_tensor'], self.samples['cache_sample_tensor']))
        shuffle(shuff_data)
        shuff_data = shuff_data[-self.max_memory_len:]
        self.samples['memory'],  self.samples['memory_tensor'], self.samples['memory_sample_tensor'] = list(zip(*shuff_data))

        shuff_data = shuff_data[-self.sample_tensor_used_len:]
        self.sample_used['sample'], self.sample_used['tensor'], self.sample_used['sample_tensor'] = list(zip(*shuff_data))
        # t3 = time()
        self.set_index(self.sample_used['tensor'])
        self.set_index(self.sample_used['sample_tensor'])
        self.samples['cache'] = []
        self.samples['cache_tensor'] = []
        self.samples['cache_sample_tensor'] = []
        self.sample_tensor_used = self.sample_used['tensor']
        # t_arr = np.array([t1, t2, t3])
        # print(t_arr[1:] - t_arr[:-1])

    def __getitem__(self, index):
        return self.sample_tensor_used[index]

    def __len__(self):
        return len(self.sample_tensor_used)


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
            'reward_tensor': reward_tensor, 'sample_index': sample_index}


def eval_gen_fn(batch_data):
    tokens, pos_ids, type_tokens, sample_index = list(zip(*batch_data))
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
    pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long).to(device)
    type_tokens_tensor = torch.tensor(type_tokens, dtype=torch.long).to(device)

    return {'input_token_tensor': tokens_tensor, 'pos_ids_tensor': pos_ids_tensor,
            'type_tokens_tensor': type_tokens_tensor, 'sample_index': sample_index}












