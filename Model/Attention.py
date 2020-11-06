import torch
import torch.nn as nn
import torch.nn.functional as F


class StatefulBilinearAttention(nn.Module):
    # query + state, key, value
    def __init__(self, query_size, state_size, key_size, hidden_size):
        super().__init__()
        self.linear_key = nn.Linear(key_size, hidden_size, bias=False)
        self.linear_query = nn.Linear(query_size, hidden_size, bias=True)
        self.linear_state = nn.Linear(state_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size = hidden_size

    def score(self, query, state, key, softmax_dim=-1, mask=None):
        w_qsk = self.matching(query, state, key, mask)
        norm_attn = F.softmax(w_qsk, dim=softmax_dim)
        if mask is not None:
            norm_attn = norm_attn.masked_fill(~mask, 0)
        return w_qsk, norm_attn

    def matching(self, query, state, key, mask=None):
        '''
        :param query: [batch_size, *, query_seq_len, query_size]
        :param key: [batch_size, *, key_seq_len, key_size]
        :param state: [batch_size, *, state_seq_len, key_size]
        :param mask: [batch_size, *, query_seq_len, key_seq_len]
        :return: [batch_size, *, query_seq_len/state_seq_len, key_seq_len]
        '''
        assert query.size(-2) == state.size(-2)  # query_seq_len == state_seq_len
        wq = self.linear_query(query)
        wq = wq.unsqueeze(-2)

        ws = self.linear_state(state)
        ws = ws.unsqueeze(-2)

        wk = self.linear_key(key)
        wk = wk.unsqueeze(-3)

        wqsk = wq + wk + ws     # [b, q_s, q_s/k_s, h]

        w_qsk = self.v(torch.tanh(wqsk)).squeeze(-1)    # [b, q_s, q_s/k_s]

        if mask is not None:
            w_qsk = w_qsk.masked_fill(~mask, -float('inf'))

        return w_qsk

    def forward(self, query, state, key, value, mask=None):
        '''
        :param query: [batch_size, *, query_seq_len, query_size]
        :param state [batch_size, *, query_seq_len, edit_size]
        :param key: [batch_size, *, key_seq_len, key_size]
        :param value: [batch_size, *, value_seq_len=key_seq_len, value_size]
        :param mask: [batch_size, *, query_seq_len, key_seq_len]
        :return: [batch_size, *, query_seq_len, value_size]
        '''

        attn, norm_attn = self.score(query, state, key, mask=mask)
        h = torch.bmm(norm_attn.view(-1, norm_attn.size(-2), norm_attn.size(-1)),
                      value.view(-1, value.size(-2), value.size(-1)))
        # attn_feature, attn, norm_attn
        return h.view(list(value.size())[:-2] + [norm_attn.size(-2), -1]), attn, norm_attn

