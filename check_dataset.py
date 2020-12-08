import torch
import argparse
from config_n import Config
from Model.pretrain_helper import get_model, get_tokenizer
from dataset.MLDDataset_HRL import MLDDatasetHRL, train_edit_gen_fn
# from Model.BertMLD import BertMLD
config = Config()


def check_epoch_0():
    from collections import Counter
    p = config.quac_dataset_path + 'bert_iter_0.train.pkl'
    data = torch.load(p)
    insert_arr = Counter()
    sub_arr = Counter()
    for d in data:
        tag_arr = d['tag']
        for tag in tag_arr:
            if tag.ope == 'I':
                insert_arr.update([len(tag.seq)])
            elif tag.ope == 'S':
                sub_arr.update([len(tag.seq)])
    insert_arr.most_common()
    sub_arr.most_common()
    # 10 is okay


def get_bert_model(args, pretrain=True):
    edit_dim = 5
    tokenizer = get_tokenizer('../extra/bert/', 'bert2bert')
    phrase_tokenizer = torch.load(config.tokenizer_path + f'bert_phrase_tokenizer.pkl')
    # model = BertMLD(model_config.decoder.hidden_size, pretrain_model, edit_dim, len(phrase_tokenizer), tokenizer,
    #                 phrase_tokenizer)
    return tokenizer, phrase_tokenizer


def check_unk_portion(args):
    test_samples = torch.load(args.data_path + f'bert_iter_0.test.pkl')
    tokenizer, phrase_tokenizer = get_bert_model(args)
    test_dataset = MLDDatasetHRL(samples=test_samples, tokenizer=tokenizer, phrase_tokenizer=phrase_tokenizer,
                                 data_type=args.mode)
    label_arr = []
    for tensor in test_dataset.sample_tensor:
        arr = [phrase_id == phrase_tokenizer.unk_token_id for phrase_id in tensor[5] if phrase_id]
        # if True in arr:
        #     print('-')
        label_arr.extend(arr)
    print(np.mean(label_arr))


def check_sample(args):
    train_samples = torch.load(args.data_path + f'bert_iter_18.train.pkl')
    tokenizer, phrase_tokenizer = get_bert_model(args)
    train_dataset = MLDDatasetHRL(samples=[], tokenizer=tokenizer, phrase_tokenizer=phrase_tokenizer,
                                  data_type=args.mode)
    train_dataset.samples = train_samples
    # train_dataset.load_sample_check()
    print('')
    def compare(arr1, arr2):
        if len(arr1) != len(arr2):
            return False
        for i in range(len(arr1)):
            if arr1[i] != arr2[i]:
                return False
        return True
    filtered_arr = []
    for tensor in train_dataset.samples['memory']:
        if not compare(tensor['current_query'], tensor['output_query']):
            filtered_arr.append(tensor)
    print(len(filtered_arr) / len(train_samples))
    print('')
    # label_arr = []
    # for tensor in test_dataset.sample_tensor:
    #     arr = [phrase_id == phrase_tokenizer.unk_token_id for phrase_id in tensor[5] if phrase_id]
    #     # if True in arr:
    #     #     print('-')
    #     label_arr.extend(arr)
    # print(np.mean(label_arr))


if __name__ == '__main__':
    import os
    import random
    import torch
    import numpy as np

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=config.quac_dataset_path)
    parser.add_argument("--mode", type=str, default='gen')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--load_epoch", type=int, default=24)
    parser.add_argument("--model_name", type=str, default="bertMLD")
    args = parser.parse_args()
    # check_epoch_0()
    # check_unk_portion(args)
    check_sample(args)

