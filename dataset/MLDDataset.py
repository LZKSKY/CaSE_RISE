from torch.utils.data import Dataset
from tqdm import tqdm
from config_n import device
from config_n import default_config as config
import torch
from copy import deepcopy
from dataset.utils import pad
from Preprocess.utils_ import Tag, obtain_gen_seq
from typing import List


class MLDDataset(Dataset):
    def __init__(self, samples=None, tokenizer=None, phrase_tokenizer=None, query_len=30, answer_len=30, n=1E10,
                 context_len=3, max_candidate_num=20, sample_tensor=None, data_type='gen'):
        super().__init__()
        self.samples = samples
        self.context_len = context_len
        self.max_candidate_num = max_candidate_num
        self.phrase_tokenizer = phrase_tokenizer
        self.max_gen_len = 20
        self.max_gen_num = 5

        if sample_tensor is None:
            self.query_len = query_len
            self.answer_len = answer_len
            self.extract_len = self.context_len * self.query_len + self.context_len * self.answer_len + 10    # 190
            self.final_len = self.extract_len + self.query_len  # 220
            self.gen_len = self.final_len + self.context_len + 1
            self.tokenizer = tokenizer
            self.edit2id = config.edit2id
            self.edit_pad_id = config.edit_pad_id
            self.n = n

            self.sample_tensor = []
            if data_type in ['gen', 'g']:
                self.load_edit_gen()
            elif data_type == 'eval':
                self.load_eval()
            else:
                raise ValueError
        else:
            self.sample_tensor = sample_tensor
            self.len = len(self.sample_tensor)

    def obtain_gen_label_v1(self, tag_arr: List[Tag], input_query):
        # seq insert
        gen_seq_arr = obtain_gen_seq(tag_arr, self.tokenizer.sep_token_id, input_query)
        seq_arr = []
        label_arr = []
        for seq in gen_seq_arr:
            decoder_seq = seq[0]
            decoder_label = [self.tokenizer.pad_token_id] * len(decoder_seq)
            decoder_seq.extend([self.tokenizer.bos_token_id] + seq[1])
            decoder_label.extend(seq[1] + [self.tokenizer.eos_token_id])
            decoder_seq = pad(decoder_seq, self.tokenizer.pad_token_id, self.max_gen_len)
            decoder_label = pad(decoder_label, self.tokenizer.pad_token_id, self.max_gen_len)
            seq_arr.append(decoder_seq)
            label_arr.append(decoder_label)
        seq_arr = pad(seq_arr, [self.tokenizer.pad_token_id] * self.max_gen_len, self.max_gen_num)
        label_arr = pad(label_arr, [self.tokenizer.pad_token_id] * self.max_gen_len, self.max_gen_num)
        return seq_arr, label_arr

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

    def load_edit_gen(self):     # edit can do with gen
        pad_token_id, sep_token_id, cls_token_id, bos_token_id, eos_token_id = \
            self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, \
            self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        for sample in tqdm(self.samples):
            context, context_answer = sample['context'][-self.context_len:], sample['context_answer'][-self.context_len:]
            input_query, output_query = sample['input_query'], sample['output_query']
            edit_tag_arr: List[Tag] = sample['tag'][:(self.query_len + 1)]
            assert len(context) == len(context_answer)
            context_token = {'input_ids': [cls_token_id], 'pos_ids': [0], 'type_ids': [0]}
            for index in range(len(context)):
                # [CLS] q [SEP] a [SEP] q [SEP] a
                seq1 = context[index][:self.query_len] + [sep_token_id]
                seq2 = context_answer[index][:self.query_len] + [sep_token_id]
                context_token['input_ids'] += seq1 + seq2
                if index == 0:
                    context_token['pos_ids'] += list(range(1, len(seq1) + 1)) + list(range(len(seq2)))
                else:
                    context_token['pos_ids'] += list(range(len(seq1))) + list(range(len(seq2)))
            context_token['input_ids'] = pad(context_token['input_ids'], pad_token_id, self.extract_len)
            context_token['pos_ids'] = pad(context_token['pos_ids'], 0, self.extract_len)
            context_token['type_ids'] = [0] * self.extract_len
            # context_token: {'input_ids', 'pos_ids', 'type_ids'} with length 190
            input_query = [bos_token_id] + input_query[:self.query_len] + [eos_token_id]
            pad_output_query = pad(output_query[:self.query_len], pad_token_id, self.query_len)
            output_tag = [self.edit2id[edit_tag.ope] for edit_tag in edit_tag_arr + [Tag('K', [])]]
            decoder_input_seq, decoder_out_seq = self.obtain_gen_label(tag_arr=edit_tag_arr, input_query=input_query)

            input_len = self.query_len + 2
            context_token['input_ids'] += pad(input_query[:self.query_len], pad_token_id, input_len)
            context_token['pos_ids'] += list(range(input_len))
            context_token['type_ids'] += [1] * input_len

            edit_pred_arr = [self.edit_pad_id] * self.extract_len + pad(output_tag, self.edit_pad_id, input_len)

            self.sample_tensor.append([context_token['input_ids'], context_token['pos_ids'], context_token['type_ids'],
                                       edit_pred_arr, decoder_input_seq, decoder_out_seq, pad_output_query])
        self.len = len(self.sample_tensor)

    def load_eval(self):
        pad_token_id, sep_token_id, cls_token_id, bos_token_id, eos_token_id = \
            self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, \
            self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        for sample in tqdm(self.samples):
            context, context_answer = sample['context'][-self.context_len:], sample['context_answer'][
                                                                             -self.context_len:]
            assert len(context) == len(context_answer)
            query = sample['query']
            groundtruth = pad(query[:self.query_len], pad_token_id, self.query_len)
            context_token = {'input_ids': [cls_token_id], 'pos_ids': [0], 'type_ids': [0]}
            for index in range(len(context)):
                # [CLS] q [SEP] a [SEP] q [SEP] a
                seq1 = context[index][:self.query_len] + [sep_token_id]
                seq2 = context_answer[index][:self.query_len] + [sep_token_id]
                context_token['input_ids'] += seq1 + seq2
                if index == 0:
                    context_token['pos_ids'] += list(range(1, len(seq1) + 1)) + list(range(len(seq2)))
                else:
                    context_token['pos_ids'] += list(range(len(seq1))) + list(range(len(seq2)))
            context_token['input_ids'] = pad(context_token['input_ids'], pad_token_id, self.extract_len)
            context_token['pos_ids'] = pad(context_token['pos_ids'], 0, self.extract_len)
            context_token['type_ids'] = [0] * self.extract_len

            cand_arr = {'input_ids': [], 'pos_ids': [], 'type_ids': []}
            for cand in sample['candidate_r']:
                pad_query = pad(cand, pad_token_id, self.query_len)
                desired_seq = {'input_ids': pad_query, 'pos_ids': list(range(len(pad_query))),
                               'type_ids': [1] * len(pad_query)}
                for k, v in desired_seq.items():
                    cand_arr[k].append(v)

            self.sample_tensor.append([context_token['input_ids'], context_token['pos_ids'], context_token['type_ids'],
                                       cand_arr['input_ids'], cand_arr['pos_ids'], cand_arr['type_ids'],
                                       groundtruth])
        self.len = len(self.sample_tensor)

    def __getitem__(self, index):
        return self.sample_tensor[index]

    def __len__(self):
        return self.len


def train_edit_gen_fn(batch_data):
    tokens, pos_ids, type_tokens, edit_labels, decoder_input, decoder_out, output_query = list(zip(*batch_data))
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
    pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long).to(device)
    type_tokens_tensor = torch.tensor(type_tokens, dtype=torch.long).to(device)
    edit_labels_tensor = torch.tensor(edit_labels, dtype=torch.long).to(device)
    decoder_input_tensor = torch.tensor(decoder_input, dtype=torch.long).to(device)
    decoder_out_tensor = torch.tensor(decoder_out, dtype=torch.long).to(device)
    output_query_tensor = torch.tensor(output_query, dtype=torch.long).to(device)

    return {'input_token_tensor': tokens_tensor, 'pos_ids_tensor': pos_ids_tensor,
            'type_tokens_tensor': type_tokens_tensor, 'edit_labels_tensor': edit_labels_tensor,
            'decoder_input_tensor': decoder_input_tensor, 'decoder_out_tensor': decoder_out_tensor,
            'query_true': output_query_tensor}


def eval_fn(batch_data):
    context_token, context_pos, context_type, cand_token, cand_pos, cand_type, groundtruth = list(zip(*batch_data))
    context_token_tensor = torch.tensor(context_token, dtype=torch.long).to(device)
    context_pos_tensor = torch.tensor(context_pos, dtype=torch.long).to(device)
    context_type_tensor = torch.tensor(context_type, dtype=torch.long).to(device)
    cand_token_tensor = torch.tensor(cand_token, dtype=torch.long).to(device)
    cand_pos_tensor = torch.tensor(cand_pos, dtype=torch.long).to(device)
    cand_type_tensor = torch.tensor(cand_type, dtype=torch.long).to(device)
    query_true_tensor = torch.tensor(groundtruth, dtype=torch.long).to(device)
    return {'context_token_tensor': context_token_tensor, 'context_pos_tensor': context_pos_tensor,
            'context_type_tensor': context_type_tensor,
            'cand_token_tensor': cand_token_tensor, 'cand_pos_tensor': cand_pos_tensor,
            'cand_type_tensor': cand_type_tensor, 'query_true_tensor': query_true_tensor}














