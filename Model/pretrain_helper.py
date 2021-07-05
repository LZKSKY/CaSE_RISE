from config_n import default_config as config
gpt2_special_tokens_dict = None
bert_special_tokens_dict = config.bert_special_tokens_dict


def get_bert_tokenizer(tokenizer_path):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    return tokenizer


def get_bert_model(model_path):
    from transformers import BertModel, BertConfig
    config = BertConfig.from_pretrained(model_path + 'config.json')
    pretrain_model = BertModel.from_pretrained(model_path + 'pytorch_model.bin', config=config)
    return pretrain_model, config


def get_bert2bert_model(model_path1, model_path2):
    from transformers import EncoderDecoderModel
    pretrain_model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_path1, model_path2)
    config = pretrain_model.config
    return pretrain_model, config


def get_model(model_path, model_name):
    if model_name == 'bert':
        tokenizer = get_bert_tokenizer(model_path)
        pretrain_model, model_config = get_bert_model(model_path)
    elif model_name == 'bert2bert':
        tokenizer = get_bert_tokenizer(model_path)
        pretrain_model, model_config = get_bert2bert_model(model_path, model_path)
    else:
        raise ValueError
    return tokenizer, pretrain_model, model_config


def get_tokenizer(model_path, model_name):
    if model_name == 'bert':
        tokenizer = get_bert_tokenizer(model_path)
    elif model_name == 'bert2bert':
        tokenizer = get_bert_tokenizer(model_path)
    else:
        raise ValueError
    return tokenizer




