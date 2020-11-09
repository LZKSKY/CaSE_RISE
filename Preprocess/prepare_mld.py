import sys
import os
import codecs
from tqdm import tqdm
import re
from config_n import Config
from Preprocess.utils_ import TagSeq
import torch

config = Config('../../')
quac_dataset_path = config.quac_dataset_path
quac_path = config.quac_path
if not os.path.exists(quac_dataset_path):
    os.mkdir(quac_dataset_path)

WORD = re.compile(r"\w+")


def load_query(file):
    query = dict()
    with codecs.open(file, encoding='utf-8') as f:
        next(f)
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t')
            if len(temp) >= 2:
                query[temp[0]] = temp[1]
    return query


def token_str(s, tokenizer):
    return tokenizer.__call__(s, add_special_tokens=False, padding='do_not_pad', truncation=True, max_length=30)['input_ids']


def tokenize_dic(dic, tokenizer):
    for k, v in dic.items():
        dic[k] = token_str(v, tokenizer)
    return dic


def load_quac_sample(answer_file, query, reformulated_query, tokenizer):   # only has reformulated query
    samples = []
    answer = dict()
    passage = dict()
    query = tokenize_dic(query, tokenizer)
    reformulated_query = tokenize_dic(reformulated_query, tokenizer)

    with codecs.open(answer_file, encoding='utf-8') as f:
        next(f)
        for line in tqdm(f):
            temp = line.strip('\n').strip('\r').split('\t')
            if len(temp) >= 2:
                history_id, q_id, pasg_id, answer_text = temp[:4]
                answer[q_id] = token_str(answer_text, tokenizer)
                passage[q_id] = pasg_id

                if len(history_id) > 1 and temp[1] in reformulated_query:
                    samp = dict()
                    samp['context_id'] = history_id.split(';')
                    samp['context_r'] = [reformulated_query.get(c_id, query[c_id]) for c_id in
                                         samp['context_id']]
                    # replace first query to its reformulated query
                    samp['context'] = [reformulated_query.get(samp['context_id'][0], query[samp['context_id'][0]])] + \
                                      [query[c_id] for c_id in samp['context_id'][1:]]
                    samp['query_id'] = q_id
                    samp['output_query'] = query[q_id]
                    samp['input_query'] = reformulated_query[q_id]
                    samp['passage_id'] = passage[q_id].split(';')
                    samples.append(samp)

    for samp in samples:
        samp['context_answer'] = [answer[q_id] for q_id in samp['context_id']]
    return samples


def load_split(file):
    train = set()
    dev = set()
    test = set()
    with codecs.open(file, encoding='utf-8') as f:
        next(f)
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t')
            if len(temp) == 2:
                if temp[1] == 'train':
                    train.add(temp[0])
                elif temp[1] == 'dev':
                    dev.add(temp[0])
                elif temp[1] == 'test':
                    test.add(temp[0])
    return train, dev, test


def split_data(split_file, samples):
    train, dev, test = load_split(split_file)
    train_samples = []
    dev_samples = []
    test_samples = []
    for sample in samples:
        if sample['query_id'] in train:
            train_samples.append(sample)
        elif sample['query_id'] in dev:
            dev_samples.append(sample)
        elif sample['query_id'] in test:
            test_samples.append(sample)
    return train_samples, dev_samples, test_samples


def check_duplicate(sample_arr):
    q_id_arr = []
    new_sample_arr = []
    for t in sample_arr:
        if t['query_id'] not in q_id_arr:
            new_sample_arr.append(t)
        q_id_arr.append(t['query_id'])
    print(f'processed before {len(sample_arr)}, processed after {len(new_sample_arr)}')
    return new_sample_arr


def extend_edit(quac_samples):
    tag_seq = TagSeq()
    for sample in quac_samples:
        sample['tag'] = tag_seq.get_label(sample['input_query'], sample['output_query'])


def build_IS_vocab(quac_samples, tokenizer):
    from collections import Counter
    c = Counter()
    sub_sig = 0     # for substitute, we can substitute for a long seq, so only count once is ok
    for sample in quac_samples:
        for tag in sample['tag']:
            if tag.ope == 'S' and sub_sig == 1:
                continue
            elif tag.ope == 'S' and sub_sig == 0:
                sub_sig = 1
            elif tag.ope != 'S':
                sub_sig = 0
            if tag.ope in ['I', 'S']:
                seq = [str(s) for s in tag.seq]
                c.update(seq)
                if ' ' in c.keys():
                    print()
                if len(tag.seq) > 1:
                    c.update([' '.join(seq)])
    pad_token_id = tokenizer.pad_token_id
    unk_token_id = tokenizer.unk_token_id
    vocab = [str(pad_token_id)] + [k for k, v in c.most_common(5000)]
    if str(unk_token_id) not in vocab:
        vocab = [str(unk_token_id)] + vocab

    str2id = dict(zip(vocab, range(1, len(vocab) + 1)))
    id2str = dict([(v, k) for k, v in str2id])
    return vocab, str2id, id2str


def preprocessing(tokenizer_path, tokenizer_name):
    from Model.pretrain_helper import get_tokenizer
    tokenizer = get_tokenizer(tokenizer_path, tokenizer_name)
    # tokenizer.add_special_tokens(bert_special_tokens_dict)
    raw_name = f'{tokenizer_name}_iter_0'
    if os.path.exists(quac_dataset_path + f'{raw_name}.train.pkl'):
        train_samples = torch.load(quac_dataset_path + f'{raw_name}.train.pkl')
        dev_samples = torch.load(quac_dataset_path + f'{raw_name}.dev.pkl')
        test_samples = torch.load(quac_dataset_path + f'{raw_name}.test.pkl')
    else:
        quac_reformulated_query = load_query(quac_path + 'quac.reformulated_query')
        quac_query = load_query(quac_path + 'quac.query')
        quac_samples = load_quac_sample(quac_path + 'quac.answer', quac_query, quac_reformulated_query, tokenizer)
        quac_samples = check_duplicate(quac_samples)
        extend_edit(quac_samples)
        train_samples, dev_samples, test_samples = split_data(quac_path + 'quac.split', quac_samples)

        # train_samples = check_duplicate(train_samples)
        # dev_samples = check_duplicate(dev_samples)
        # test_samples = check_duplicate(test_samples)
        print('data size', len(train_samples), len(dev_samples), len(test_samples))
        torch.save(train_samples, quac_dataset_path + f'{raw_name}.train.pkl')
        torch.save(dev_samples, quac_dataset_path + f'{raw_name}.dev.pkl')
        torch.save(test_samples, quac_dataset_path + f'{raw_name}.test.pkl')
    # extend_edit(train_samples)
    build_IS_vocab(train_samples, tokenizer)
    # extend_edit(dev_samples)
    # extend_edit(test_samples)
    # torch.save(train_samples, quac_dataset_path + f'{raw_name}.train.pkl')
    # torch.save(dev_samples, quac_dataset_path + f'{raw_name}.dev.pkl')
    # torch.save(test_samples, quac_dataset_path + f'{raw_name}.test.pkl')
    print('processed quac dataset')


# def preprocessing_find(tokenizer_path, tokenizer_name):
#     from Model.pretrain_helper import get_tokenizer
#     tokenizer = get_tokenizer(tokenizer_path, tokenizer_name)
#     # tokenizer.add_special_tokens(bert_special_tokens_dict)
#     raw_name = f'{tokenizer_name}_raw'
#     # train_samples = torch.load(quac_dataset_path + f'{raw_name}.train.pkl')
#     # dev_samples = torch.load(quac_dataset_path + f'{raw_name}.dev.pkl')
#     test_samples = torch.load(quac_dataset_path + f'{raw_name}.test.pkl')
#
#     test_samples = extend_edit(test_samples, pad_id=tokenizer.pad_token_id)
#     torch.save(test_samples, quac_dataset_path + f'{raw_name}.test.pkl')
#
#     print('process quac dataset')
#
#     # train_samples = check_duplicate(train_samples)
#     # dev_samples = check_duplicate(dev_samples)
#     # test_samples = check_duplicate(test_samples)
#     # # for t in ['gen', 'rank', 'edit']:
#     # need_list = list(zip(['train', "dev", 'test'], [train_samples, dev_samples, test_samples]))
#     # for t in ['rank', 'edit_gen', 'eval'][:2]:
#     #     for s, samples in need_list[:]:
#     #         dataset = Bert2BertDataset(samples=samples, tokenizer=tokenizer, data_type=t)
#     #         torch.save(dataset.sample_tensor[:100], quac_dataset_path + f'{tokenizer_name}.{s}.{t}.dataset.pkl')
#
#     # new_test_samples = []
#     # arr = []
#     # for t in test_samples:
#     #     if t['query_id'] not in arr:
#     #         new_test_samples.append(t)
#     #     arr.append(t['query_id'])
#     # dataset = CaSEDataset(samples=new_test_samples, tokenizer=tokenizer, data_type='eval')
#     # torch.save(dataset.sample_tensor, quac_dataset_path + f'case.eval.dataset.pkl')


if __name__ == '__main__':
    preprocessing('../../extra/bert/', 'bert')
    # preprocessing_find('../../extra/bert/', 'bert')

















