import torch

class PhraseTokenizer:
    def __init__(self, c, bert_tokenizer):
        pad_token = str(bert_tokenizer.pad_token_id)
        unk_token = str(bert_tokenizer.unk_token_id)
        vocab = [pad_token] + [k for k, v in c.most_common(5000)]
        if unk_token not in vocab:
            vocab = [unk_token] + vocab
        self.vocab = vocab
        self.str2id = dict(zip(vocab, range(1, len(vocab) + 1)))
        self.id2str = dict([(v, k) for k, v in self.str2id.items()])
        self.pad_token_id = self.str2id[pad_token]
        self.unk_token_id = self.str2id[unk_token]
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.len = len(vocab) + 1

    def __len__(self):
        return self.len

    def convert_token_to_id(self, token_arr):
        if isinstance(token_arr, list):
            token_str = ' '.join([str(t) for t in token_arr])
        else:
            token_str = str(token_arr)
        return self.str2id.get(token_str, self.unk_token_id)

    def convert_id_to_token_arr(self, seq_id):
        w_arr = self.id2str.get(seq_id, self.unk_token)
        w_arr = [int(w) for w in w_arr.split(' ')]
        return w_arr

    def convert_id_arr_to_token_arr(self, seq_id_arr):
        if isinstance(seq_id_arr, torch.Tensor):
            seq_id_arr = seq_id_arr.detach().cpu().numpy()
        token_arr = []
        for seq_id in seq_id_arr:
            token_arr.append(self.convert_id_to_token_arr(seq_id))
        return token_arr





