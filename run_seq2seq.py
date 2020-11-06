from Model.Bert2Bert import Bert2Bert
from Model.Bert2BertEdit import Bert2BertEdit
from Model.Bert2BertFinetune import Bert2BertFinetune
from Model.pretrain_helper import get_model
from config import quac_dataset_path, output_path
from dataset.Dataset_bert import Bert2BertDataset, train_rank_fn, train_edit_fn, train_gen_fn, train_edit_gen_fn, eval_fn
import torch
from Trainer.BertTrainer import BertTrainer
import argparse


def get_bert_model(args, pretrain=True):
    edit_dim = 4
    tokenizer, pretrain_model, model_config = get_model('../extra/bert/', 'bert2bert')
    require_grad = pretrain is not True
    for name, para in pretrain_model.named_parameters():
        if 'cls' in name:
            para.requires_grad = True
        else:
            para.requires_grad = require_grad
    if args.model_name == 'bert2bert':
        model = Bert2Bert(model_config.decoder.hidden_size, pretrain_model, edit_dim, model_config.decoder.vocab_size, tokenizer)
    elif args.model_name == 'bert2bertEdit':
        model = Bert2BertEdit(model_config.decoder.hidden_size, pretrain_model, edit_dim, model_config.decoder.vocab_size, tokenizer)
    else:
        model = Bert2BertFinetune(model_config.decoder.hidden_size, pretrain_model, edit_dim, model_config.decoder.vocab_size, tokenizer, model_config.decoder)
    return model


def run_bert_per(args):
    edit_dim = 4
    max_epoch = 50
    # batch_size = 12     # 16 for rank, 16 for edit
    batch_size = args.batch_size    # 16 for rank, 16 for edit
    accumulation_steps = args.accumulation_steps
    print('loading dataset')
    # mode = 'gen'
    mode = args.mode
    model_name = '-'.join([args.model_name, mode])
    collate = {'rank': train_rank_fn, 'edit': train_edit_fn, 'gen': train_gen_fn, 'edit_gen': train_edit_gen_fn}[mode]
    train_samples = torch.load(args.data_path + f'bert.train.{mode}.dataset.pkl')[:10000]
    dev_samples = torch.load(args.data_path + f'bert.dev.{mode}.dataset.pkl')[:1000]
    samples = train_samples + dev_samples
    train_size = len(samples)
    print('data size', train_size)

    train_dataset = Bert2BertDataset(sample_tensor=train_samples, data_type=mode)
    dev_dataset = Bert2BertDataset(sample_tensor=dev_samples, data_type=mode)

    model = get_bert_model(args)
    trainer = BertTrainer(model=model, batch_size=batch_size, accumulation_steps=accumulation_steps, model_name=model_name,
                          ema_rate=0.995, max_epoch=25, initial_lr=5e-5, tune_lr=5e-5, tune_epoch=5, train_size=train_size)
    trainer.train_type = mode
    trainer.train(train_dataset, collate, dev_dataset, collate, max_epoch=max_epoch)


def run_bert_raw(args):
    from collections import defaultdict
    edit_dim = 4
    max_epoch = 20
    # batch_size = 12     # 16 for rank, 16 for edit
    train_arr = defaultdict(lambda: {})
    train_size = 0
    print('loading dataset')
    for m in ['rank', 'edit_gen']:
        train_num = {'rank': 100000, 'edit_gen': 10000}
        dev_num = {'rank': 10000, 'edit_gen': 1000}
        print(f'loading {m}')
        collate = {'rank': train_rank_fn, 'edit': train_edit_fn, 'gen': train_gen_fn, 'edit_gen': train_edit_gen_fn}[m]
        train_samples = torch.load(args.data_path + f'bert.train.{m}.dataset.pkl')[:train_num[m]]
        dev_samples = torch.load(args.data_path + f'bert.dev.{m}.dataset.pkl')[:dev_num[m]]
        print(f'{m} train samples len {len(train_samples)}, dev samples len {len(dev_samples)}')
        train_dataset = Bert2BertDataset(sample_tensor=train_samples, data_type=m)
        dev_dataset = Bert2BertDataset(sample_tensor=dev_samples, data_type=m)
        train_size += len(train_samples) + len(dev_samples)
        train_arr['train_samples'][m] = train_samples
        train_arr['dev_samples'][m] = dev_samples
        train_arr['train_dataset'][m] = train_dataset
        train_arr['dev_dataset'][m] = dev_dataset
        train_arr['train_collate'][m] = train_arr['dev_collate'][m] = collate
    train_arr['batch_size'] = {'rank': 16, 'edit': 16, 'gen': 12, 'edit_gen': 16}
    train_arr['accumulation'] = {'rank': 4, 'edit': 2, 'gen': 3, 'edit_gen': 4}
    model = get_bert_model(args)
    model_name = args.model_name + '-all'
    trainer = BertTrainer(model=model, batch_size=-1, accumulation_steps=-1, model_name=model_name,
                          ema_rate=0.995, max_epoch=max_epoch, initial_lr=5e-5, tune_lr=5e-5, tune_epoch=5, train_size=train_size)
    trainer.train_bert(train_arr)


def run_bert_gen(args):
    edit_dim = 4
    batch_size = args.batch_size  # 16 for rank, 16 for edit
    accumulation_steps = args.accumulation_steps
    print('loading dataset')
    mode = 'edit_gen'
    collate = train_edit_gen_fn
    train_samples = torch.load(args.data_path + f'bert.test.{mode}.dataset.pkl')[:1000]
    # dev_samples = torch.load(args.data_path + f'bert.dev.{mode}.dataset.pkl')[:1000]
    samples = train_samples
    train_size = len(samples)
    print('data size', train_size)

    train_dataset = Bert2BertDataset(sample_tensor=train_samples, data_type=mode)

    model = get_bert_model(args)
    # model = Bert2Bert(model_config.hidden_size, pretrain_model, edit_dim, model_config.vocab_size, tokenizer)
    model_name = args.model_name + '-all'
    trainer = BertTrainer(model=model, batch_size=batch_size, accumulation_steps=accumulation_steps,
                          # model_name=f'bert2bertEdit-all',
                          model_name=model_name,
                          ema_rate=0.995, max_epoch=25, initial_lr=5e-5, tune_lr=5e-5, tune_epoch=5,
                          train_size=train_size)
    trainer.train_type = mode
    trainer.load_model(args.load_epoch)
    trainer.generate_seq(train_dataset, collate, f'{output_path}bert2bert-{args.load_epoch}_gen_out.pkl')


def run_bert_eval(args):
    edit_dim = 4
    batch_size = args.batch_size  # 16 for rank, 16 for edit
    accumulation_steps = args.accumulation_steps
    print('loading dataset')
    mode = 'eval'
    test_samples = torch.load(args.data_path + f'bert.test.{mode}.dataset.pkl')
    print(f'test samples {len(test_samples)}')

    test_dataset = Bert2BertDataset(sample_tensor=test_samples, data_type=mode)

    model = get_bert_model(args)
    # model = Bert2Bert(model_config.hidden_size, pretrain_model, edit_dim, model_config.vocab_size, tokenizer)
    model_name = args.model_name + '-all'
    trainer = BertTrainer(model=model, batch_size=batch_size, accumulation_steps=accumulation_steps,
                          model_name=model_name,
                          ema_rate=0.995, max_epoch=25, initial_lr=5e-5, tune_lr=5e-5, tune_epoch=5,
                          train_size=-1)
    trainer.load_model(args.load_epoch)     # 10
    trainer.eval_bert(test_dataset, eval_fn, f'{output_path}{model_name}-{args.load_epoch}_eval_out.pkl')


def run_bert_check_loss(args):
    from collections import defaultdict
    edit_dim = 4
    max_epoch = 50
    # batch_size = 12     # 16 for rank, 16 for edit
    train_arr = defaultdict(lambda: {})
    train_size = 0
    print('loading dataset')
    for m in ['rank', 'edit_gen']:
        print(f'loading {m}')
        collate = {'rank': train_rank_fn, 'edit': train_edit_fn, 'gen': train_gen_fn, 'edit_gen': train_edit_gen_fn}[m]
        train_samples = torch.load(args.data_path + f'bert.train.{m}.dataset.pkl')
        dev_samples = torch.load(args.data_path + f'bert.dev.{m}.dataset.pkl')
        print(f'{m} train samples len {len(train_samples)}, dev samples len {len(dev_samples)}')
        train_dataset = Bert2BertDataset(sample_tensor=train_samples, data_type=m)
        dev_dataset = Bert2BertDataset(sample_tensor=dev_samples, data_type=m)
        train_size += len(train_samples) + len(dev_samples)
        train_arr['train_samples'][m] = train_samples
        train_arr['dev_samples'][m] = dev_samples
        train_arr['train_dataset'][m] = train_dataset
        train_arr['dev_dataset'][m] = dev_dataset
        train_arr['train_collate'][m] = train_arr['dev_collate'][m] = collate
    train_arr['batch_size'] = {'rank': 64, 'edit': 16, 'gen': 12, 'edit_gen': 64}
    train_arr['accumulation'] = {'rank': 4, 'edit': 2, 'gen': 3, 'edit_gen': 4}
    model = get_bert_model(args)

    trainer = BertTrainer(model=model, batch_size=-1, accumulation_steps=-1, model_name=f'{args.model_name}-all',
                          ema_rate=0.995, max_epoch=max_epoch, initial_lr=5e-5, tune_lr=5e-5, tune_epoch=5, train_size=train_size)
    trainer.check_loss(train_arr)


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
    parser.add_argument("--data_path", type=str, default=quac_dataset_path)
    parser.add_argument("--mode", type=str, default='edit_gen')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--load_epoch", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="bert2bertEditFinetune")
    args = parser.parse_args()
    # run_bert_gen(args)
    run_bert_per(args)
    # run_bert_raw(args)
    # run_bert_eval(args)
    # run_bert_check_loss(args)
