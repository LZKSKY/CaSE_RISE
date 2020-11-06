import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import EncoderDecoderModel
min_num = 1e-8
from Model.seq2seq_gen_helper import generate


class Bert2Bert(nn.Module):
    def __init__(self, hidden_size, pretrain_model: EncoderDecoderModel, edit_voc_dim, gen_voc_dim, tokenizer):
        super().__init__()
        self.hidden_size = hidden_size
        self.pretrain_model = pretrain_model
        self.rank_linear = nn.Sequential(nn.Linear(self.hidden_size, 256), nn.GELU(), nn.Linear(256, 1))
        # self.pretrain_model._init_weights(self.rank_linear[0])
        # self.pretrain_model._init_weights(self.rank_linear[2])
        self.edit_linear = nn.Sequential(nn.Linear(self.hidden_size, 256), nn.GELU(), nn.Linear(256, edit_voc_dim))
        self.generate_linear = nn.Linear(self.hidden_size, gen_voc_dim)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.edit2id = {'': 0, 'c': 1, 'd': 2, 'i': 3}

    def do_rank(self, data, method=None):
        '''
        :param data:
        :param method:
        :return:
        '''
        query_token = data['input_token_tensor']
        pos_token = data['pos_ids_tensor']
        rank_label = data['rank_labels_tensor']
        input_ids = query_token
        input_pos_ids = pos_token
        hidden = self.pretrain_model.encoder.__call__(
            input_ids=input_ids,
            attention_mask=input_ids != self.pad_token_id,
            position_ids=input_pos_ids
        )[0]

        rank_output = self.rank_linear(hidden[:, 0]).squeeze(dim=-1)
        weight = torch.ones_like(rank_output)
        if method == 'train':
            loss = F.binary_cross_entropy_with_logits(rank_output, rank_label.float(), reduction='mean', weight=weight)
            return rank_output, loss
        elif method == 'eval':
            rank_output_sigmoid = (rank_output.sigmoid() >= 0.5).float()
            return {'output': rank_output_sigmoid, 'label': rank_label}

    def do_edit_gen_raw(self, data, method=None):
        input_ids = data['input_token_tensor']
        input_pos_ids = data['pos_ids_tensor']
        type_tokens = data['type_tokens_tensor']
        edit_labels = data['edit_labels_tensor']
        decoder_edit = data['decoder_edit_tensor']
        decoder_word = data['decoder_word_tensor']
        dist_label = data['dist_label_tensor']
        decoder_input_ids = decoder_word[:, :-1]
        decoder_output_ids = decoder_word[:, 1:]
        # all_hidden [b, s, hidden]
        hidden = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=input_ids != self.pad_token_id,
            position_ids=input_pos_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_input_ids != self.pad_token_id,
            labels=decoder_output_ids,
            return_dict=True
        )
        if method == 'train':
            edit_out = self.edit_linear(hidden.encoder_last_hidden_state)
            edit_loss = F.nll_loss(edit_out.log_softmax(dim=-1).transpose(1, 2), edit_labels, ignore_index=0, reduction='mean')
            gen_loss = F.nll_loss(hidden.logits.log_softmax(dim=-1).transpose(1, 2), decoder_output_ids, ignore_index=self.pad_token_id, reduction='mean')
            return (edit_out, hidden.logits), (edit_loss, gen_loss)
        elif method == 'eval':
            edit_out = self.edit_linear(hidden.encoder_last_hidden_state)
            edit_loss = F.nll_loss(edit_out.log_softmax(dim=-1).transpose(1, 2), edit_labels, ignore_index=0,
                                   reduction='mean')
            gen_out = hidden.logits
            nll_loss = F.nll_loss(hidden.logits.log_softmax(dim=-1).transpose(1, 2), decoder_output_ids, ignore_index=self.pad_token_id, reduction='mean')
            return {'edit_output': edit_out, 'edit_label': edit_labels, 'edit_loss': edit_loss,
                    'gen_out': gen_out, 'gen_label': decoder_output_ids, 'gen_loss': nll_loss}

    def do_edit_pred(self):
        raise NotImplementedError

    def do_gen(self, data, method=None):
        raise NotImplementedError

    def generate(self, data, method=None):  # only for test generation
        input_ids = data['input_token_tensor']
        input_pos_ids = data['pos_ids_tensor']
        output_ids = generate(self.pretrain_model, input_ids, pos_ids=input_pos_ids, max_length=30,
                              min_length=0, do_sample=False, early_stopping=True,
                              num_beams=1, bos_token_id=self.bos_token_id, pad_token_id=self.pad_token_id,
                              eos_token_id=self.eos_token_id, decoder_start_token_id=self.bos_token_id)
        return output_ids

    def rank_args(self, input_ids, input_pos_ids):
        hidden = self.pretrain_model.encoder.__call__(
            input_ids=input_ids,
            attention_mask=input_ids != self.pad_token_id,
            position_ids=input_pos_ids
        )[0]
        rank_output = self.rank_linear(hidden[:, 0]).squeeze(dim=-1)    # [c]
        return rank_output

    def batch_rank(self, *args, **kwargs):
        input_ids = args[0]
        input_pos_ids = args[1]
        max_batch_size = 20
        cand_size = input_ids.size(0)
        rank_out_arr = []
        for index_i in range(0, cand_size, max_batch_size):
            index_j = min(cand_size, max_batch_size + index_i)
            rank_out_batch = self.rank_args(input_ids[index_i:index_j], input_pos_ids[index_i:index_j])
            rank_out_arr.append(rank_out_batch)
        rank_out = torch.cat(rank_out_arr, dim=0) if len(rank_out_arr) > 1 else rank_out_arr[0]
        return rank_out

    def ids2str(self, sentence, skip_special_tokens=True):
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(sentence.cpu().numpy(), skip_special_tokens=skip_special_tokens))

    def rank_generate(self, data, method=None):
        context_ids = data['context_token_tensor']      # [1, s1]
        context_pos_ids = data['context_pos_tensor']
        assert context_ids.size(0) == 1
        cand_token = data['cand_token_tensor'].squeeze(dim=0)      # [1, c, s2] -> [c, s2]
        cand_pos_ids = data['cand_pos_tensor'].squeeze(dim=0)
        cand_num = cand_token.size(0)
        input_ids = torch.cat((context_ids.expand(cand_num, -1), cand_token), dim=1)    # [c, s1 + s2]
        input_pos_ids = torch.cat((context_pos_ids.expand(cand_num, -1), cand_pos_ids), dim=1)
        rank_out = self.batch_rank(input_ids, input_pos_ids)
        rank_index = rank_out.argmax(dim=0).cpu().item()

        output_ids = generate(self.pretrain_model, input_ids[rank_index].unsqueeze(dim=0),
                              pos_ids=input_pos_ids[rank_index].unsqueeze(dim=0), max_length=30,
                              min_length=0, do_sample=False, early_stopping=True,
                              num_beams=1, bos_token_id=self.bos_token_id, pad_token_id=self.pad_token_id,
                              eos_token_id=self.eos_token_id, decoder_start_token_id=self.bos_token_id)
        output_str = self.ids2str(output_ids[0])
        output_str_true = self.ids2str(data['query_true_tensor'][0])
        sel_query_true = self.ids2str(data['cand_token_tensor'][0][0])
        context_l = torch.sum((context_ids[0] == self.pad_token_id).float(), dim=0).item()
        context_str = self.ids2str(context_ids[0][:int(context_ids.size(1) - context_l)], skip_special_tokens=False)
        sel_query = self.ids2str(cand_token[rank_index])

        return {'rank_out': rank_out.cpu().numpy(), 'output_ids': output_str, 'query_true': output_str_true,
                'context': context_str, 'sel_query': sel_query, 'sel_query_true': sel_query_true}

    def forward(self, data, method='train'):
        if method == 'dev':
            return self.do_dev(data)
        elif method == 'train':
            return self.do_train(data)
        elif method == 'test':
            return self.do_test(data)
