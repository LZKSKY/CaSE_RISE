import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import EncoderDecoderModel
from config_n import Config
import numpy as np
from Model.utils import pad
from Model.utils import merge_path
from Preprocess.PhraseTokenizer import PhraseTokenizer
from typing import Dict
config = Config()
min_num = 1e-8


class BertMLD(nn.Module):
    def __init__(self, hidden_size, pretrain_model: EncoderDecoderModel, edit_voc_dim, gen_voc_dim, tokenizer,
                 phrase_tokenizer: PhraseTokenizer):
        super().__init__()
        self.hidden_size = hidden_size
        self.pretrain_model = pretrain_model
        self.edit_linear = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.GELU(),
                                         nn.LayerNorm(self.hidden_size, eps=pretrain_model.decoder.config.layer_norm_eps),
                                         nn.Linear(self.hidden_size, edit_voc_dim))
        self.gen_linear = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.GELU(),
                                        nn.LayerNorm(self.hidden_size, eps=pretrain_model.decoder.config.layer_norm_eps),
                                        nn.Linear(self.hidden_size, gen_voc_dim))
        self.score_linear = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.GELU(),
                                          nn.LayerNorm(self.hidden_size,
                                                       eps=pretrain_model.decoder.config.layer_norm_eps),
                                          nn.Linear(self.hidden_size, 1))
        self.tokenizer = tokenizer
        self.max_pred_len = 10
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.edit2id = config.edit2id
        self.id2edit = config.id2edit
        self.phrase_tokenizer = phrase_tokenizer

    def edit_pred(self, data, method=None, compute_loss=True):
        input_ids = data['input_token_tensor']  # [b, s1]
        input_pos_ids = data['pos_ids_tensor']
        type_tokens = data['type_tokens_tensor']
        encoder_attention_mask = input_ids != self.pad_token_id
        _, encoder_seq_len = input_ids.size()
        # encoder_hidden [b, s, hidden]
        encoder_output = self.pretrain_model.encoder(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            position_ids=input_pos_ids,
            token_type_ids=type_tokens,
            return_dict=True
        )
        edit_out = self.edit_linear(encoder_output.last_hidden_state)
        if compute_loss:
            edit_labels = data['edit_labels_tensor']  # [b, s1]
            reward = data['reward_tensor']
            if method == 'rl_train':
                edit_loss = F.nll_loss(edit_out.log_softmax(dim=-1).transpose(1, 2), edit_labels, ignore_index=0,
                                       reduction='none')    # [b, s]
                edit_loss = edit_loss.sum(dim=1) / (edit_labels > 0).float().sum(dim=1)
                edit_loss = (edit_loss * reward).mean()
            elif method == 'prob':
                edit_loss = F.nll_loss(edit_out.log_softmax(dim=-1).transpose(1, 2), edit_labels, ignore_index=0,
                                       reduction='none')  # [b, s]
                edit_loss = edit_loss.sum(dim=1) / (edit_labels > 0).float().sum(dim=1)     # [b]
            else:
                edit_loss = F.nll_loss(edit_out.log_softmax(dim=-1).transpose(1, 2), edit_labels, ignore_index=0,
                                       reduction='mean')
            return {'edit_output': edit_out, 'edit_loss': edit_loss, 'encoder_output': encoder_output}
        else:
            return {'edit_output': edit_out, 'encoder_output': encoder_output}

    def seq_predict(self, data, encoder_output, compute_loss=True, method=None):
        input_ids = data['input_token_tensor']  # [b, s1]
        decoder_input = data['decoder_input_tensor']  # [b, l, s2]

        encoder_attention_mask = input_ids != self.pad_token_id
        decoder_attention_mask = decoder_input != self.pad_token_id
        batch_size, gen_dim, decoder_seq_len = decoder_input.size()
        encoder_seq_len = encoder_output[0].size(1)
        encoder_hidden_expand = encoder_output[0].unsqueeze(dim=1).expand(batch_size, gen_dim, encoder_seq_len,
                                                                          -1).contiguous(). \
            view(batch_size * gen_dim, encoder_seq_len, -1)  # [b * l, s1]
        encoder_attention_mask_expand = encoder_attention_mask.unsqueeze(dim=1).expand(-1, gen_dim, -1).contiguous(). \
            view(batch_size * gen_dim, encoder_seq_len)  # [b * l, s1]
        decoder_input_expand = decoder_input.contiguous().view(batch_size * gen_dim, decoder_seq_len)
        decoder_attention_mask_expand = decoder_attention_mask.contiguous().view(batch_size * gen_dim, decoder_seq_len)

        decoder_outputs = self.pretrain_model.decoder(
            input_ids=decoder_input_expand,
            inputs_embeds=None,
            attention_mask=decoder_attention_mask_expand,
            encoder_hidden_states=encoder_hidden_expand,
            encoder_attention_mask=encoder_attention_mask_expand,
            return_dict=True,
            output_hidden_states=True,
        )
        gen_hidden = decoder_outputs.hidden_states[-1][:, -1]  # [b * s, V]
        gen_out = self.gen_linear(gen_hidden)
        gen_out_view = gen_out.contiguous().view(batch_size, gen_dim, -1)
        if compute_loss:
            decoder_label = data['decoder_out_tensor']  # [b, l, s2]
            decoder_label_expand = decoder_label.contiguous().view(-1)
            if method == 'rl_train':
                reward = data['reward_tensor']  # [b]
                gen_loss = F.nll_loss(gen_out.log_softmax(dim=-1), decoder_label_expand, ignore_index=0,
                                      reduction='none')
                gen_loss = reward.unsqueeze(dim=1).expand(-1, gen_dim).contiguous().view(-1) * gen_loss
                gen_loss = gen_loss.sum(dim=0) / ((decoder_label_expand > 0).float().sum(dim=0) + 1)  # []
            elif method == 'prob':
                gen_loss = F.nll_loss(gen_out.log_softmax(dim=-1), decoder_label_expand, ignore_index=0,
                                      reduction='none')
                # [b]
                gen_loss = gen_loss.contiguous().view(-1, gen_dim).sum(dim=1) / ((decoder_label > 0).float().sum(dim=1) + 1)
            else:
                gen_loss = F.nll_loss(gen_out.log_softmax(dim=-1), decoder_label_expand, ignore_index=0,
                                      reduction='mean')     # []
            return {'gen_output': gen_out_view, 'gen_loss': gen_loss}
        else:
            return {'gen_output': gen_out_view}

    def do_edit_gen(self, data, method='train', compute_loss=True, **kwargs):
        edit_dict = self.edit_pred(data, compute_loss=compute_loss, method=method)
        gen_dict: Dict = self.seq_predict(data, edit_dict['encoder_output'], compute_loss=compute_loss, method=method)

        decoder_label = data['decoder_out_tensor']  # [b, l, s2]
        edit_labels = data['edit_labels_tensor']  # [b, s1]
        decoder_label_expand = decoder_label.contiguous().view(-1)
        reward = data['reward_tensor']
        if config.use_stopping_score:
            score_dict: Dict = self.score_predict(edit_dict['encoder_output'], compute_loss=True, reward=reward)

        gen_dict.update(edit_dict)
        if config.use_stopping_score:
            gen_dict.update(score_dict)
        gen_dict.update({'edit_label': edit_labels, 'gen_label': decoder_label_expand, 'reward': reward})
        return gen_dict

    def forward(self, data, method='train', compute_loss=True, **kwargs):
        return self.do_edit_gen(data, method,  compute_loss=compute_loss, **kwargs)

    def edit2sequence(self, edit_out: torch.Tensor, input_ids: torch.Tensor, encoder_attention_mask: torch.Tensor):
        edit_out_prob = edit_out.softmax(dim=-1)
        edit_out_prob[:, :, 0] = 0.
        edit_label = edit_out_prob.argmax(dim=-1)
        edit_label[~encoder_attention_mask] = 0
        edit_label[input_ids == self.eos_token_id] = self.edit2id['K']
        edit_label_numpy = edit_label.detach().cpu().numpy()
        input_ids_numpy = input_ids.detach().cpu().numpy()
        seq_left_arr = []
        seq2gen_arr = []
        for index_i, seq in enumerate(input_ids_numpy):
            seq_left_i = []
            seq_gen_i = []
            index_j = 0
            seq_l_ij = np.sum(seq != self.pad_token_id)
            edit_path = merge_path(edit_label_numpy[index_i])
            while index_j < seq_l_ij:
                edit_ij = self.id2edit[edit_path[index_j]]
                if edit_ij == 'K':
                    seq_left_i.append(seq[index_j])
                    index_j += 1
                elif edit_ij == 'D':
                    index_j += 1
                elif edit_ij == 'I':
                    seq_gen_i.append(pad([seq[index_j], seq[index_j + 1], self.cls_token_id],
                                         self.pad_token_id, self.max_pred_len, padding_mode='l'))
                    index_j += 1
                    seq_left_i.append('<pad>')
                elif edit_ij == 'S':
                    temp_arr = [seq[index_j]]
                    index_j += 1
                    while index_j < seq_l_ij and self.id2edit[edit_path[index_j]] == 'S':
                        temp_arr.append(seq[index_j])
                        index_j += 1
                    seq_gen_i.append(pad(temp_arr + [self.tokenizer.cls_token_id], self.pad_token_id, self.max_pred_len, padding_mode='l'))
                    seq_left_i.append('<pad>')
                else:
                    break
            seq2gen_arr.append(seq_gen_i)
            seq_left_arr.append(seq_left_i)
        max_insert_len = max([len(s) for s in seq2gen_arr])
        if max_insert_len == 0:
            return seq_left_arr, []
        else:
            seq2gen_arr = [pad(s, [self.pad_token_id] * self.max_pred_len, max_insert_len) for s in seq2gen_arr]
            return seq_left_arr, seq2gen_arr,

    def gen_final_ids(self, seq_left_arr, output_ids):
        insert_ids = output_ids
        final_seqs = []
        for index_i, seq in enumerate(seq_left_arr):
            temp_seq = []
            insert_seq_count = 0
            for index_j, w in enumerate(seq):
                if w == '<pad>':
                    if insert_seq_count < len(insert_ids[index_i]):
                        temp_seq.extend(insert_ids[index_i][insert_seq_count])
                        insert_seq_count += 1
                else:
                    temp_seq.append(w)
            final_seqs.append(temp_seq)
        return final_seqs

    def ids2str(self, sentence, skip_special_tokens=True):
        if isinstance(sentence, torch.Tensor):
            new_s = sentence.cpu().numpy()
        else:
            new_s = sentence
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(new_s, skip_special_tokens=skip_special_tokens))

    def generate_edit_gen(self, data, method=None):
        input_ids = data['input_token_tensor']  # [b, s1]
        encoder_attention_mask: torch.Tensor = input_ids != self.pad_token_id
        _, encoder_seq_len = input_ids.size()
        # encoder_hidden [b, s, hidden]
        edit_dict = self.edit_pred(data, method, compute_loss=False)
        edit_out, encoder_output = edit_dict['edit_output'], edit_dict['encoder_output']
        # [b, l, s], [b, s]
        seq_left_arr, seq2gen_arr = self.edit2sequence(edit_out[:, 190:], input_ids[:, 190:], encoder_attention_mask[:, 190:])
        if seq2gen_arr:
            # =================  gen insert ids ========================================
            seq2gen_tensor = torch.tensor(seq2gen_arr, dtype=input_ids.dtype, device=input_ids.device)
            gen_data = {'input_token_tensor': input_ids, 'decoder_input_tensor': seq2gen_tensor}
            gen_dict = self.seq_predict(gen_data, encoder_output, compute_loss=False)
            gen_out = gen_dict['gen_output']
            gen_id = gen_out.argmax(dim=-1)     # [b, max_s]
            output_ids = self.phrase_tokenizer.convert_id_arr_to_token_arr(gen_id.contiguous().view(-1))
            # [b, max_insert, seq]
            insert_l = gen_out.size(1)
            # print(insert_l, len(output_ids), len(seq_left_arr))
            output_ids = [output_ids[index:(index + insert_l)] for index in range(0, len(output_ids), insert_l)]
            # ==================   gen final ids ======================================
            final_seqs = self.gen_final_ids(seq_left_arr, output_ids)  # [b, s]
        else:
            # ==================   gen final ids ======================================
            final_seqs = seq_left_arr  # [b, s]

        if config.use_stopping_score:
            score_dict = self.score_predict(encoder_output, compute_loss=False)
            score = score_dict['score_output'].sigmoid().detach().cpu().numpy()

        arr = []
        for index in range(len(input_ids)):
            keys = ['input_query', 'gen_query', 'gen_query_ids', 'edit_ids']
            seqs = (self.ids2str(input_ids[index, 190:]), self.ids2str(final_seqs[index]), final_seqs[index][1:-1],
                    edit_out[:, 190:].softmax(dim=-1).argmax(dim=-1)[index])
            final_dict = dict(zip(keys, seqs))
            if config.use_stopping_score:
                final_dict.update({'stopping_score': score[index]})
            arr.append(final_dict)

        return arr

