from Model.BertMLD import BertMLD
from Model.pretrain_helper import get_model
from config_n import Config
from dataset.MLDDataset_HRL import MLDDatasetHRL, train_edit_gen_fn, eval_gen_fn
import torch
from Trainer.RLTrainer import RLTrainer
import argparse
config = Config()


def get_bert_model(args, pretrain=True):
    edit_dim = 5
    tokenizer, pretrain_model, model_config = get_model('../extra/bert/', 'bert2bert')
    require_grad = pretrain is not True
    for name, para in pretrain_model.named_parameters():
        # para.requires_grad = True
        if 'cls' in name:
            para.requires_grad = True
        else:
            para.requires_grad = require_grad
    phrase_tokenizer = torch.load(config.tokenizer_path + f'bert_phrase_tokenizer.pkl')
    model = BertMLD(model_config.decoder.hidden_size, pretrain_model, edit_dim, len(phrase_tokenizer), tokenizer,
                    phrase_tokenizer)
    return model, tokenizer, phrase_tokenizer


def check_sample(args):
    print('loading dataset')
    mode = args.mode
    model_name = '-'.join([args.model_name, mode])

    model, tokenizer, phrase_tokenizer = get_bert_model(args)
    train_samples = torch.load(args.data_path + f'bert_iter_20.train.pkl')
    train_samples['samples'] = train_samples['samples'][:100]
    train_samples['memory'] = train_samples['memory'][:100]

    train_dataset = MLDDatasetHRL(samples=[], tokenizer=tokenizer, phrase_tokenizer=phrase_tokenizer,
                                  data_type=args.mode)
    train_dataset.samples = train_samples
    train_dataset.load_sample_check()

    trainer = RLTrainer(model=model, batch_size=config.batch_size, accumulation_steps=config.accumulation_steps,
                        model_name=model_name, ema_rate=0.995, max_epoch=config.max_epoch * 2,
                        initial_lr=config.initial_lr, tune_lr=config.tune_lr, tune_epoch=config.tune_epoch,
                        train_size=200, load_epoch=args.load_epoch, train_dataset=train_dataset,
                        train_collate=train_edit_gen_fn)
    trainer.set_save_path(model_name=model_name + '-RL')
    trainer.load_model(20)
    trainer.sample_step(trainer.dataset, trainer.collate_fn, max_count=10, verbose=False, fold=-1)


def check_sample_statistics(args):
    print('loading dataset')
    mode = args.mode
    model_name = '-'.join([args.model_name, mode])

    model, tokenizer, phrase_tokenizer = get_bert_model(args)
    train_samples = torch.load(args.data_path + f'bert_iter_20.train.pkl')
    # train_samples['samples'] = train_samples['samples'][:100]
    # train_samples['memory'] = train_samples['memory'][:100]

    train_dataset = MLDDatasetHRL(samples=[], tokenizer=tokenizer, phrase_tokenizer=phrase_tokenizer,
                                  data_type=args.mode)
    train_dataset.samples = train_samples
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
        # if not compare(tensor['current_query'], tensor['output_query']):      # 0.96
        if not compare(tensor['current_query'], tensor['input_query']):         # 0.88
            filtered_arr.append(tensor)

    print(len(filtered_arr) / len(train_samples['memory']))
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
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--accumulation_steps", type=int, default=config.accumulation_steps)
    parser.add_argument("--load_epoch", type=int, default=config.load_epoch)
    parser.add_argument("--model_name", type=str, default="bertMLD")
    args = parser.parse_args()
    # check_epoch_0()
    # check_unk_portion(args)
    # check_sample(args)
    check_sample_statistics(args)

