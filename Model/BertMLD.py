import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import EncoderDecoderModel
from config_n import Config
import numpy as np
from Model.utils import pad
from Model.MLD_gen_helper import generate
from Model.utils import merge_path
from Preprocess.PhraseTokenizer import PhraseTokenizer
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
        self.tokenizer = tokenizer
        self.max_pred_len = 10
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
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
            return edit_out, edit_loss, encoder_output
        else:
            return edit_out, encoder_output

    def seq_predict(self, data, encoder_output, compute_loss=True, method=None):
        input_ids = data['input_token_tensor']  # [b, s1]
        decoder_input = data['decoder_input_tensor']  # [b, l, s2]
        decoder_label = data['decoder_out_tensor']  # [b, l, s2]
        encoder_attention_mask = input_ids != self.pad_token_id
        decoder_attention_mask = decoder_input != self.pad_token_id
        batch_size, gen_dim, decoder_seq_len = decoder_input.size()
        _, encoder_seq_len = input_ids.size()
        encoder_hidden_expand = encoder_output[0].unsqueeze(dim=1).expand(batch_size, gen_dim, encoder_seq_len,
                                                                          -1).contiguous(). \
            view(batch_size * gen_dim, encoder_seq_len, -1)  # [b * l, s1]
        encoder_attention_mask_expand = encoder_attention_mask.unsqueeze(dim=1).expand(-1, gen_dim, -1).contiguous(). \
            view(batch_size * gen_dim, encoder_seq_len)  # [b * l, s1]
        decoder_input_expand = decoder_input.contiguous().view(batch_size * gen_dim, decoder_seq_len)
        decoder_attention_mask_expand = decoder_attention_mask.contiguous().view(batch_size * gen_dim, decoder_seq_len)
        decoder_label_expand = decoder_label.contiguous().view(-1)

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
        if compute_loss:
            if method == 'rl_train':
                reward = data['reward_tensor']  # [b]
                gen_loss = F.nll_loss(gen_out.log_softmax(dim=-1), decoder_label_expand, ignore_index=0,
                                      reduction='none')
                gen_loss = reward.unsqueeze(dim=1).expand(-1, gen_dim).contiguous().view(-1) * gen_loss
                gen_loss = gen_loss.sum(dim=0) / (decoder_label_expand > 0).float().sum(dim=0)  # []
            elif method == 'prob':
                gen_loss = F.nll_loss(gen_out.log_softmax(dim=-1), decoder_label_expand, ignore_index=0,
                                      reduction='none')
                # [b]
                gen_loss = gen_loss.contiguous().view(-1, gen_dim).sum(dim=1) / (decoder_label > 0).float().sum(dim=1)
            else:
                gen_loss = F.nll_loss(gen_out.log_softmax(dim=-1), decoder_label_expand, ignore_index=0,
                                      reduction='mean')     # []
            return gen_out, gen_loss
        else:
            return gen_out

    def do_edit_gen(self, data, method=None):
        edit_out, edit_loss, encoder_output = self.edit_pred(data, method)
        gen_out, gen_loss = self.seq_predict(data, encoder_output, compute_loss=True, method=method)
        if method == 'train':
            return (edit_out, gen_out), (edit_loss, gen_loss)
        elif method == 'eval':
            decoder_label = data['decoder_out_tensor']  # [b, l, s2]
            edit_labels = data['edit_labels_tensor']  # [b, s1]
            decoder_label_expand = decoder_label.contiguous().view(-1)
            return {'edit_output': edit_out, 'edit_label': edit_labels, 'edit_loss': edit_loss,
                    'gen_out': gen_out, 'gen_label': decoder_label_expand, 'gen_loss': gen_loss}

    def edit2sequence_v1(self, edit_out: torch.Tensor, input_ids: torch.Tensor, encoder_attention_mask: torch.Tensor):
        # let SDS = SSS; 但是预测出来还是烂的
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
            while index_j < seq_l_ij:
                edit_ij = self.id2edit[edit_label_numpy[index_i][index_j]]
                if edit_ij == 'K':
                    seq_left_i.append(seq[index_j])
                    index_j += 1
                elif edit_ij == 'D':
                    index_j += 1
                elif edit_ij == 'I':
                    seq_gen_i.append(pad([seq[index_j], self.tokenizer.sep_token_id,  seq[index_j + 1]],
                                         self.pad_token_id, 5, padding_mode='l') + [self.bos_token_id])
                    index_j += 1
                    seq_left_i.append('<pad>')
                elif edit_ij == 'S':
                    temp_arr = [seq[index_j]]
                    index_j += 1
                    while index_j < seq_l_ij and self.id2edit[edit_label_numpy[index_i][index_j]] == 'S':
                        temp_arr.append(seq[index_j])
                        index_j += 1
                    seq_gen_i.append(pad(temp_arr, self.pad_token_id, 5, padding_mode='l') + [self.bos_token_id])
                    seq_left_i.append('<pad>')
                else:
                    break
            seq2gen_arr.append(seq_gen_i)
            seq_left_arr.append(seq_left_i)
        max_insert_len = max([len(s) for s in seq2gen_arr])
        seq2gen_arr = [pad(s, [self.pad_token_id] * 6, max_insert_len) for s in seq2gen_arr]
        return seq_left_arr, seq2gen_arr,

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
        seq2gen_arr = [pad(s, [self.pad_token_id] * self.max_pred_len, max_insert_len) for s in seq2gen_arr]
        return seq_left_arr, seq2gen_arr,

    def gen_final_ids_v1(self, seq_left_arr, output_ids):
        output_ids_numpy = output_ids.detach().cpu().numpy()[:, 6:]
        insert_ids = [[i for i in ids if i not in self.tokenizer.all_special_ids] for ids in output_ids_numpy]
        final_seqs = []
        for index_i, seq in enumerate(seq_left_arr):
            temp_seq = []
            insert_seq_count = 0
            for w in seq:
                if w == '<pad>':
                    temp_seq.extend(insert_ids[index_i][insert_seq_count])
                    insert_seq_count += 1
                else:
                    temp_seq.append(w)
            final_seqs.append(temp_seq)
        return final_seqs

    def gen_final_ids(self, seq_left_arr, output_ids):
        insert_ids = output_ids
        final_seqs = []
        for index_i, seq in enumerate(seq_left_arr):
            temp_seq = []
            insert_seq_count = 0
            for w in seq:
                if w == '<pad>':
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

    def generate_edit_gen_v1(self, data, method=None):
        input_ids = data['input_token_tensor']  # [b, s1]
        encoder_attention_mask = input_ids != self.pad_token_id
        _, encoder_seq_len = input_ids.size()
        # encoder_hidden [b, s, hidden]
        edit_out, edit_loss, encoder_output = self.edit_pred(data, method)
        # [b, l, s], [b, s]
        seq_left_arr, seq2gen_arr = self.edit2sequence(edit_out[:, 190:], input_ids[:, 190:], encoder_attention_mask[:, 190:])
        # =================  gen insert ids ========================================
        seq2gen_tensor = torch.tensor(seq2gen_arr, dtype=input_ids.dtype, device=input_ids.device)
        insert_l = seq2gen_tensor.size(1)
        batch_size, in_seq_len, hidden_size = encoder_output.last_hidden_state.size()
        encoder_output.last_hidden_state = encoder_output.last_hidden_state.unsqueeze(dim=1).\
            expand(batch_size, insert_l, in_seq_len, hidden_size).contiguous().view(-1, in_seq_len, hidden_size)
        seq2gen_tensor = seq2gen_tensor.view(batch_size * insert_l, -1)
        encoder_attention_mask_expand = encoder_attention_mask.unsqueeze(dim=1).\
            expand(batch_size, insert_l, -1).contiguous().view(batch_size * insert_l, -1)
        output_ids = generate(self.pretrain_model, input_ids=seq2gen_tensor, encoder_outputs=encoder_output,
                              max_length=20, attention_mask=encoder_attention_mask_expand)
        # ==================   gen final ids ======================================
        final_seqs = self.gen_final_ids(seq_left_arr, output_ids)
        arr = []
        for index in range(len(input_ids)):
            keys = ['input_query', 'gen_query', 'output_query', 'edit_ids']
            seqs = (self.ids2str(input_ids[index, 190:]), self.ids2str(final_seqs[index]),
                    self.ids2str(data['query_true'][index]), edit_out[:, 190:].softmax(dim=-1).argmax(dim=-1)[index])
            arr.append(dict(zip(keys, seqs)))
        return arr

    def generate_edit_gen(self, data, method=None):
        input_ids = data['input_token_tensor']  # [b, s1]
        encoder_attention_mask = input_ids != self.pad_token_id
        _, encoder_seq_len = input_ids.size()
        # encoder_hidden [b, s, hidden]
        edit_out, encoder_output = self.edit_pred(data, method, compute_loss=False)
        # [b, l, s], [b, s]
        seq_left_arr, seq2gen_arr = self.edit2sequence(edit_out[:, 190:], input_ids[:, 190:], encoder_attention_mask[:, 190:])
        # =================  gen insert ids ========================================
        seq2gen_tensor = torch.tensor(seq2gen_arr, dtype=input_ids.dtype, device=input_ids.device)
        insert_l = seq2gen_tensor.size(1)
        batch_size, in_seq_len, hidden_size = encoder_output.last_hidden_state.size()
        seq2gen_tensor = seq2gen_tensor.view(batch_size * insert_l, -1)
        encoder_attention_mask_expand = encoder_attention_mask.unsqueeze(dim=1).\
            expand(batch_size, insert_l, -1).contiguous().view(batch_size * insert_l, -1)
        encoder_hidden_expand = encoder_output[0].unsqueeze(dim=1).expand(batch_size, insert_l, encoder_seq_len, -1).\
            contiguous().view(batch_size * insert_l, encoder_seq_len, -1)  # [b * l, s1]
        decoder_outputs = self.pretrain_model.decoder(
            input_ids=seq2gen_tensor,
            inputs_embeds=None,
            attention_mask=seq2gen_tensor != self.pad_token_id,
            encoder_hidden_states=encoder_hidden_expand,
            encoder_attention_mask=encoder_attention_mask_expand,
            return_dict=True,
            output_hidden_states=True,
        )
        gen_hidden = decoder_outputs.hidden_states[-1][:, -1]  # [b * s, V]
        gen_out = self.gen_linear(gen_hidden).argmax(dim=-1)   # [b * insert_l, 1]
        output_ids = self.phrase_tokenizer.convert_id_arr_to_token_arr(gen_out.contiguous().view(-1))
        output_ids = [output_ids[index:(index + insert_l)] for index in range(0, len(output_ids), insert_l)]
        # ==================   gen final ids ======================================
        final_seqs = self.gen_final_ids(seq_left_arr, output_ids)
        arr = []
        for index in range(len(input_ids)):
            keys = ['input_query', 'gen_query', 'output_query', 'edit_ids']
            seqs = (self.ids2str(input_ids[index, 190:]), self.ids2str(final_seqs[index]),
                    self.ids2str(data['query_true'][index]), edit_out[:, 190:].softmax(dim=-1).argmax(dim=-1)[index])
            arr.append(dict(zip(keys, seqs)))
        return arr

    def forward(self, data, method='train'):
        return self.do_edit_gen(data, method)
