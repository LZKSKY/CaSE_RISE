from Model.BertMLD import BertMLD
from Model.pretrain_helper import get_model
from config_n import Config
from dataset.MLDDataset_HRL import MLDDatasetHRL, train_edit_gen_fn, eval_gen_fn
import torch
from Trainer.RLTrainer import RLTrainer
import argparse
config = Config()
quac_dataset_path, output_path = config.quac_dataset_path, config.output_path


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


def run_bert_train(args):
    batch_size = args.batch_size    # 16 for rank, 16 for edit
    accumulation_steps = args.accumulation_steps
    print('loading dataset')
    mode = args.mode
    model_name = '-'.join([args.model_name, mode])

    # dev_samples = torch.load(args.data_path + f'bert_iter_0.dev.pkl')

    model, tokenizer, phrase_tokenizer = get_bert_model(args)

    if config.load_epoch <= 10:
        train_samples = torch.load(args.data_path + f'bert_iter_0.train.pkl')[:1000]
        train_dataset = MLDDatasetHRL(samples=train_samples, tokenizer=tokenizer, phrase_tokenizer=phrase_tokenizer,
                                      data_type=mode)
        # train_dataset.load_sample_check()
    else:
        sampling_strategy = config.sampling_strategy
        train_samples = torch.load(args.data_path + f'bert_iter_{config.load_epoch}.{sampling_strategy}.pkl')
        train_dataset = MLDDatasetHRL(samples=train_samples['origin'], tokenizer=tokenizer, phrase_tokenizer=phrase_tokenizer,
                                      data_type=mode)
        # train_samples['origin'] = train_samples['origin'][:10000]
        train_dataset.samples = train_samples
    train_size = len(train_dataset.samples['origin'])
    print('train_size', train_size)
    trainer = RLTrainer(model=model, batch_size=batch_size, accumulation_steps=accumulation_steps,
                        model_name=model_name, ema_rate=0.995, max_epoch=config.max_epoch * 2,
                        initial_lr=config.initial_lr, tune_lr=config.tune_lr, tune_epoch=config.tune_epoch,
                        train_size=train_size, load_epoch=args.load_epoch, train_dataset=train_dataset,
                        train_collate=train_edit_gen_fn, sampling_strategy=config.sampling_strategy)
    trainer.set_save_path(model_name=model_name + '-RL')
    print('loaded model')
    trainer.train_rl()


def run_bert_generate(args):
    from Evaluation.evaluation import compute_score
    batch_size = args.batch_size    # 16 for rank, 16 for edit
    accumulation_steps = args.accumulation_steps
    print('loading dataset')
    mode = args.mode
    model_name = '-'.join([args.model_name, mode])
    # train_samples = torch.load(args.data_path + f'bert_iter_0.train.pkl')
    test_samples = torch.load(args.data_path + f'bert_iter_0.test.pkl')
    samples = test_samples
    train_size = len(samples)
    print('data size', train_size)

    model, tokenizer, phrase_tokenizer = get_bert_model(args)
    dev_dataset = MLDDatasetHRL(samples=test_samples, tokenizer=tokenizer, phrase_tokenizer=phrase_tokenizer,
                                data_type=mode)
    dev_dataset.load_sample_gen()
    trainer = RLTrainer(model=model, batch_size=batch_size, accumulation_steps=accumulation_steps,
                        model_name=model_name, ema_rate=0.995, max_epoch=config.max_epoch * 2,
                        initial_lr=config.initial_lr, tune_lr=config.tune_lr, tune_epoch=config.tune_epoch,
                        train_size=train_size, load_epoch=args.load_epoch)
    trainer.set_save_path(model_name=model_name + '-RL')
    print('load model')
    # for load_epoch in range(config.rl_begin_epoch + 1, config.rl_epoch):
    for load_epoch in range(21, 40):
        trainer.load_model(load_epoch)
        save_f_name = f'{config.output_path}bert_mld-{load_epoch}_gen_out.pkl'
        trainer.generate_mld(dev_dataset, eval_gen_fn, save_f_name)
        compute_score(save_f_name, list(range(1, config.L + 1)))


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
    parser.add_argument("--mode", type=str, default='gen')
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--accumulation_steps", type=int, default=config.accumulation_steps)
    parser.add_argument("--load_epoch", type=int, default=config.load_epoch)
    parser.add_argument("--model_name", type=str, default="bertMLD")
    args = parser.parse_args()
    run_bert_train(args)
    # run_bert_generate(args)

    # TODO: check reward；为啥出来的action都一样, 一开始内存用的是全部的



