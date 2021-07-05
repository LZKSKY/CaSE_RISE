import os
import re
import torch
from copy import deepcopy
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from dataset.json_helper import load_json
from config_n import Config
from Preprocess.utils_ import TagSeq

config = Config('../')
quac_dataset_path = config.quac_dataset_path
quac_path = config.quac_path
if not os.path.exists(quac_dataset_path):
    os.mkdir(quac_dataset_path)

WORD = re.compile(r"\w+")


def token_str(s, tokenizer):
    return tokenizer.__call__(s, add_special_tokens=False, padding='do_not_pad', truncation=True, max_length=30)['input_ids']


def build_samples(conv_arr, rewrite_arr, tokenizer):
    rewrite_dict = dict(rewrite_arr)
    info_arr = []
    for conv in conv_arr:
        context = []
        conv_id = conv['number']
        for i, obj in enumerate(conv['turn']):
            sample = {}
            turn_id = obj['number']
            sample['output_query'] = token_str(obj['raw_utterance'], tokenizer)
            sample['input_query'] = rewrite_dict['_'.join([str(conv_id), str(turn_id)])]
            sample['input_query'] = token_str(sample['input_query'], tokenizer)
            sample['context'] = deepcopy(context)
            sample['context_answer'] = [[]] * len(sample['context'])
            if i == 0:
                context.append(sample['input_query'])
            else:
                context.append(sample['output_query'])
            info_arr.append(sample)
    return info_arr


def preprocessing(tokenizer_path, tokenizer_name):
    from Model.pretrain_helper import get_tokenizer
    tokenizer = get_tokenizer(tokenizer_path, tokenizer_name)
    data_path = '../../CAsT/'
    raw_name = f'{tokenizer_name}_iter_0'
    conv_file = data_path + 'evaluation_topics_v1.0.json'
    rewrite_file = data_path + 'evaluation_topics_annotated_resolved_v1.0.tsv'
    conv_arr = load_json(conv_file)
    rewrite_arr = []
    with open(rewrite_file, 'r', encoding='utf-8') as f:
        for line in f:
            rewrite_arr.append(line.rstrip('\n').split('\t'))
    sample_arr = build_samples(conv_arr, rewrite_arr, tokenizer)
    torch.save(sample_arr, quac_dataset_path + f'{raw_name}.CAsT_test.pkl')
    print('processed CAsT dataset')


if __name__ == '__main__':
    preprocessing('../../extra/bert/', 'bert')

















