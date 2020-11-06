from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from config import bert_special_tokens_dict as special_tokens_dict, device
import torch
from copy import deepcopy


def pad(ori_arr, pad_value, desired_num):
    assert desired_num > 0
    result = ori_arr[:desired_num] + [pad_value] * (desired_num - len(ori_arr))
    assert len(result) == desired_num
    return result


class Bert2BertDataset(Dataset):
    def __init__(self, samples=None, tokenizer=None, query_len=30, answer_len=30, n=1E10, context_len=3,
                 max_candidate_num=20, sample_tensor=None, data_type='rank'):
        super().__init__()
        self.samples = samples
        self.context_len = context_len
        self.max_candidate_num = max_candidate_num

        if sample_tensor is None:
            self.query_len = query_len
            self.answer_len = answer_len
            self.extract_len = self.context_len * self.query_len + self.context_len * self.answer_len + 10    # 190
            self.final_len = self.extract_len + self.query_len  # 220
            self.gen_len = self.final_len + self.context_len + 1
            self.tokenizer = tokenizer
            # pad_token_id = self.tokenizer.pad_token_id
            pad_word = 0
            self.edit2id = {0: 0, 'K': 1, 'I': 2, 'D': 3, 'S': 4}
            self.n = n

            self.sample_tensor = []
            if data_type == 'rank' or data_type == 'r':
                self.load_rank()
            elif data_type == 'edit' or data_type == 'e':
                self.load_edit()
            elif data_type == 'gen' or data_type == 'g':
                self.load_edit_gen()
            elif data_type == 'edit_gen' or data_type == 'eg':
                self.load_edit_gen()
            elif data_type == 'eval':
                self.load_eval()
            else:
                raise ValueError
        else:
            self.sample_tensor = sample_tensor
            self.len = len(self.sample_tensor)

    def load_rank(self):
        pad_token_id, sep_token_id, cls_token_id, bos_token_id, eos_token_id = \
            self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, \
            self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        for sample in tqdm(self.samples):
            context, context_answer = sample['context'][-self.context_len:], sample['context_answer'][-self.context_len:]
            assert len(context) == len(context_answer)
            context_token = {'input_ids': [cls_token_id], 'pos_ids': [0], 'type_ids': [0]}
            # for position id in index 1,                   [0, 1, 2, 3, ...] means position
            # for type if in index 2,                       [0, 1, 2] 0 means padding, 1 means question, 2 means answer
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
            for index, cand in enumerate(sample['candidate_r']):
                desired_seq = [
                    context_token['input_ids'] + pad(cand, pad_token_id, self.query_len),
                    context_token['pos_ids'] + list(range(self.query_len)),
                    context_token['type_ids'] + [1] * self.query_len
                ]
                self.sample_tensor.append(desired_seq + [0, int(index == 0)])
        self.len = len(self.sample_tensor)

    def load_edit(self):
        pad_token_id, sep_token_id, cls_token_id, bos_token_id, eos_token_id = \
            self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, \
            self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        for sample in tqdm(self.samples):
            context, context_answer = sample['context'][-self.context_len:], sample['context_answer'][-self.context_len:]
            query_r, query_r_edit = sample['query_r'], sample['query_r_edit']
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
            edit_pred_arr = [0] * self.extract_len + [self.edit2id.get(edit[0], 0) for edit in query_r_edit[:self.query_len]]
            edit_pred_arr = pad(edit_pred_arr, 0, self.final_len)
            context_token['input_ids'] += pad(query_r[:self.query_len], pad_token_id, self.query_len)
            context_token['pos_ids'] += list(range(self.query_len))
            context_token['type_ids'] += [1] * self.query_len
            self.sample_tensor.append([context_token['input_ids'], context_token['pos_ids'], context_token['type_ids'], edit_pred_arr])
        self.len = len(self.sample_tensor)

    def load_edit_gen(self):     # edit can do with gen
        pad_token_id, sep_token_id, cls_token_id, bos_token_id, eos_token_id = \
            self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, \
            self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        for sample in tqdm(self.samples):
            context, context_answer = sample['context'][-self.context_len:], sample['context_answer'][-self.context_len:]
            query_r, query_r_edit = sample['query_r'], sample['query_r_edit']
            query_edit = sample['query_edit'][:self.query_len]
            dist_supervision = sample['distance_supervision']
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
            edit_pred_arr = [0] * self.extract_len + [self.edit2id.get(edit[0], 0) for edit in query_r_edit[:self.query_len]]
            edit_pred_arr = pad(edit_pred_arr, 0, self.final_len)
            context_token['input_ids'] += pad(query_r[:self.query_len], pad_token_id, self.query_len)
            context_token['pos_ids'] += list(range(self.query_len))
            context_token['type_ids'] += [1] * self.query_len

            # ============
            decoder_edit_arr = [1] + [self.edit2id.get(edit[0], 0) for edit in query_edit] + [1]    # [q + 2]
            decoder_word_arr = [bos_token_id] + [edit[1] for edit in query_edit] + [eos_token_id]    # [q + 2]
            dist_sup_arr = dist_supervision + [[pad_token_id] * 4]  # [q + 1]
            decoder_edit_arr = pad(decoder_edit_arr, 0, self.query_len + 2)
            decoder_word_arr = pad(decoder_word_arr, pad_token_id, self.query_len + 2)
            dist_sup_arr = pad(dist_sup_arr, [pad_token_id] * 4, self.query_len + 1)

            self.sample_tensor.append([context_token['input_ids'], context_token['pos_ids'], context_token['type_ids'],
                                       edit_pred_arr, decoder_edit_arr, decoder_word_arr, dist_sup_arr])
        self.len = len(self.sample_tensor)

    def load_gen(self):
        pad_token_id, sep_token_id, cls_token_id, bos_token_id, eos_token_id, unk_token_id = \
            self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, \
            self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.eos_token_id
        for sample in tqdm(self.samples):
            context, context_answer = sample['context'][-self.context_len:], sample['context_answer'][-self.context_len:]
            query_r, query_edit = sample['query_r'][:self.query_len], sample['query_edit'][:self.query_len]
            dist_supervision = sample['distance_supervision']
            assert len(context) == len(context_answer)
            context_token = [cls_token_id]
            for index in range(len(context)):
                # [CLS] q [SEP] a [SEP] q [SEP] a
                context_token += context[index][:self.query_len] + [sep_token_id] + \
                                 context_answer[index][:self.query_len] + [sep_token_id]
            context_token += query_r[:self.query_len]
            output_query = [edit[1] for edit in query_edit]
            input_token_arr = context_token + [bos_token_id] + output_query
            output_token_arr = output_query + [eos_token_id]

            edit_label_arr = [self.edit2id.get(edit[0], 0) for edit in query_edit] + [0]
            dist_sup_arr = dist_supervision + [[pad_token_id] * 4]

            pad_input_token_arr = pad(input_token_arr, pad_token_id, self.gen_len)
            pad_edit_label_arr = pad(edit_label_arr, 0, self.query_len + 1)
            pad_output_token_arr = pad(output_token_arr, pad_token_id, self.query_len + 1)
            pad_dist_sup_arr = pad(dist_sup_arr, [pad_token_id] * 4, self.query_len + 1)

            mask_v, un_mask_v = 0, 1
            mask_context = [un_mask_v] * len(context_token)
            mask_arr = []
            label_index_arr = []
            for index in range(1, len(output_query) + 2):
                # first bos
                mask_arr.append(pad(mask_context + [un_mask_v] * index, mask_v, self.gen_len))
                label_index_arr.append(len(context_token) + index - 1)    # hope to use nll loss to extract value
            pad_mask_arr = pad(mask_arr, [mask_v] * self.gen_len, self.query_len + 1)
            pad_label_index_arr = pad(label_index_arr, -1, self.query_len + 1)
            self.sample_tensor.append([pad_input_token_arr, pad_output_token_arr, pad_edit_label_arr, pad_dist_sup_arr,
                                       pad_mask_arr, pad_label_index_arr])
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


def train_rank_fn(batch_data):
    tokens, pos_ids, token_type, sel_index, labels = list(zip(*batch_data))
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
    pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long).to(device)
    type_tensor = torch.tensor(token_type, dtype=torch.long).to(device)
    sel_index_tensor = torch.tensor(sel_index, dtype=torch.long).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    return {'input_token_tensor': tokens_tensor,
            'pos_ids_tensor': pos_ids_tensor,
            'type_tensor': type_tensor,
            'sel_index_tensor': sel_index_tensor,
            'rank_labels_tensor': labels_tensor}


def train_edit_gen_fn(batch_data):
    tokens, pos_ids, type_tokens, edit_labels, decoder_edit, decoder_word, dist_label = list(zip(*batch_data))
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
    pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long).to(device)
    type_tokens_tensor = torch.tensor(type_tokens, dtype=torch.long).to(device)
    edit_labels_tensor = torch.tensor(edit_labels, dtype=torch.long).to(device)
    decoder_edit_tensor = torch.tensor(decoder_edit, dtype=torch.long).to(device)
    decoder_word_tensor = torch.tensor(decoder_word, dtype=torch.long).to(device)
    dist_label_tensor = torch.tensor(dist_label, dtype=torch.long).to(device)

    return {'input_token_tensor': tokens_tensor, 'pos_ids_tensor': pos_ids_tensor,
            'type_tokens_tensor': type_tokens_tensor, 'edit_labels_tensor': edit_labels_tensor,
            'decoder_edit_tensor': decoder_edit_tensor, 'decoder_word_tensor': decoder_word_tensor,
            'dist_label_tensor': dist_label_tensor}


def train_edit_fn(batch_data):
    tokens, pos_ids, type_tokens, edit_labels = list(zip(*batch_data))
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
    pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long).to(device)
    type_tokens_tensor = torch.tensor(type_tokens, dtype=torch.long).to(device)
    edit_labels_tensor = torch.tensor(edit_labels, dtype=torch.long).to(device)
    return {'input_token_tensor': tokens_tensor, 'pos_ids_tensor': pos_ids_tensor,
            'type_tokens_tensor': type_tokens_tensor, 'edit_labels_tensor': edit_labels_tensor}


def train_gen_fn(batch_data):
    tokens, pad_edit_input_arr, output_tokens, edit_labels, dist_labels, mask_arr, label_index, query_r_id, query_r_mask = list(zip(*batch_data))
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
    pad_edit_input_tensor = torch.tensor(pad_edit_input_arr, dtype=torch.long).to(device)
    output_tokens_tensor = torch.tensor(output_tokens, dtype=torch.long).to(device)
    edit_labels_tensor = torch.tensor(edit_labels, dtype=torch.long).to(device)
    dist_labels_tensor = torch.tensor(dist_labels, dtype=torch.long).to(device)
    mask_arr_tensor = torch.tensor(mask_arr, dtype=torch.long).to(device)
    label_index_tensor = torch.tensor(label_index, dtype=torch.long).to(device)
    query_r_id = torch.tensor(query_r_id, dtype=torch.long).to(device)
    query_r_mask = torch.tensor(query_r_mask, dtype=torch.long).to(device)
    return {'input_token_tensor': tokens_tensor, 'pad_edit_input_tensor': pad_edit_input_tensor,
            'output_tokens_tensor': output_tokens_tensor,
            'edit_labels_tensor': edit_labels_tensor, 'dist_labels_tensor': dist_labels_tensor,
            'mask_arr_tensor': mask_arr_tensor, 'label_index_tensor': label_index_tensor,
            'query_r_id': query_r_id, 'query_r_mask': query_r_mask}


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














