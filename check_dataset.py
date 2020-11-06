import torch
from config_n import Config
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


if __name__ == '__main__':
    check_epoch_0()


