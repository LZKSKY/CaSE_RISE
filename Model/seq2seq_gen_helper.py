from typing import Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from config import logger_model as logger
from transformers.file_utils import ModelOutput
from transformers import EncoderDecoderModel


def generate(
        seq2seq_model: EncoderDecoderModel,
        input_ids: Optional[torch.LongTensor] = None,
        pos_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs
) -> torch.LongTensor:
    r"""
    Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
    beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

    Adapted in part from `Facebook's XLM beam search code
    <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

    Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
    attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
    indicated are the default values of those config.

    Most of these parameters are explained in more detail in `this blog post
    <https://huggingface.co/blog/how-to-generate>`__.

    Parameters:

        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            The sequence used as a prompt for the generation. If :obj:`None` the method initializes
            it as an empty :obj:`torch.LongTensor` of shape :obj:`(1,)`.
        max_length (:obj:`int`, `optional`, defaults to 20):
            The maximum length of the sequence to be generated.
        min_length (:obj:`int`, `optional`, defaults to 10):
            The minimum length of the sequence to be generated.
        do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beams (:obj:`int`, `optional`, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        temperature (:obj:`float`, `optional`, defaults tp 1.0):
            The value used to module the next token probabilities.
        top_k (:obj:`int`, `optional`, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (:obj:`float`, `optional`, defaults to 1.0):
            If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or
            higher are kept for generation.
        repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        bos_token_id (:obj:`int`, `optional`):
            The id of the `beginning-of-sequence` token.
        eos_token_id (:obj:`int`, `optional`):
            The id of the `end-of-sequence` token.
        length_penalty (:obj:`float`, `optional`, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty.

            Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
            order to encourage the model to produce longer sequences.
        no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
            If set to int > 0, all ngrams of that size can only occur once.
        bad_words_ids(:obj:`List[int]`, `optional`):
            List of token ids that are not allowed to be generated. In order to get the tokens of the words that
            should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
        num_return_sequences(:obj:`int`, `optional`, defaults to 1):
            The number of independently computed returned sequences for each element in the batch.
        attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
            tokens that are not masked, and 0 for masked tokens.

            If not provided, will default to a tensor the same shape as :obj:`input_ids` that masks the pad token.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_start_token_id (:obj:`int`, `optional`):
            If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
        use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.

    Return:

        :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`:
        The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
        shorter if all batches finished early due to the :obj:`eos_token_id`.

    Examples::

        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
        outputs = model.generate(max_length=40)  # do greedy decoding
        print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
        input_context = 'The dog'
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
        for i in range(3): #  3 output sequences were generated
            print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
        input_context = 'The dog'
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True)  # generate 3 candidates using sampling
        for i in range(3): #  3 output sequences were generated
            print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

        tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
        input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
        print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
        input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
        bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
    """

    # We cannot generate if the model does not have a LM head
    if seq2seq_model.get_output_embeddings() is None:
        raise AttributeError(
            "You tried to generate sequences with a model that does not have a LM Head."
            "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
        )

    max_length = max_length if max_length is not None else seq2seq_model.config.max_length
    min_length = min_length if min_length is not None else seq2seq_model.config.min_length
    do_sample = do_sample if do_sample is not None else seq2seq_model.config.do_sample
    early_stopping = early_stopping if early_stopping is not None else seq2seq_model.config.early_stopping
    use_cache = use_cache if use_cache is not None else seq2seq_model.config.use_cache
    num_beams = num_beams if num_beams is not None else seq2seq_model.config.num_beams
    temperature = temperature if temperature is not None else seq2seq_model.config.temperature
    top_k = top_k if top_k is not None else seq2seq_model.config.top_k
    top_p = top_p if top_p is not None else seq2seq_model.config.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else seq2seq_model.config.repetition_penalty
    bos_token_id = bos_token_id if bos_token_id is not None else seq2seq_model.config.bos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else seq2seq_model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else seq2seq_model.config.eos_token_id
    length_penalty = length_penalty if length_penalty is not None else seq2seq_model.config.length_penalty
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else seq2seq_model.config.no_repeat_ngram_size
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else seq2seq_model.config.bad_words_ids
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else seq2seq_model.config.num_return_sequences
    )
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else seq2seq_model.config.decoder_start_token_id
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = 1

    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
    assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
    assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
    assert temperature > 0, "`temperature` should be strictly positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
    ), "`pad_token_id` should be a positive integer."
    assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
    ), "`eos_token_id` should be a positive integer."
    assert length_penalty > 0, "`length_penalty` should be strictly positive."
    assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
    ), "`no_repeat_ngram_size` should be a positive integer."
    assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictly positive integer."
    assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
    ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

    if input_ids is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
            "you should either supply a context to complete as `input_ids` input "
            "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
        )
        input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=next(seq2seq_model.parameters()).device,
        )
    else:
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    # not allow to duplicate outputs when greedy decoding
    if do_sample is False:
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                    num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                    num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

    # create attention mask if necessary
    # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
    if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    # set pad_token_id to eos_token_id if not set. Important that this is done after
    # attention_mask is created
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(
            "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
        )
        pad_token_id = eos_token_id

    # vocab size
    if hasattr(seq2seq_model.config, "vocab_size"):
        vocab_size = seq2seq_model.config.vocab_size
    elif (
            seq2seq_model.config.is_encoder_decoder
            and hasattr(seq2seq_model.config, "decoder")
            and hasattr(seq2seq_model.config.decoder, "vocab_size")
    ):
        vocab_size = seq2seq_model.config.decoder.vocab_size
    else:
        raise ValueError("either seq2seq_model.config.vocab_size or seq2seq_model.config.decoder.vocab_size needs to be defined")

    # set effective batch size and effective batch multiplier according to do_sample
    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1

    if seq2seq_model.config.is_encoder_decoder:
        if decoder_start_token_id is None:
            # see if BOS token can be used for decoder_start_token_id
            if bos_token_id is not None:
                decoder_start_token_id = bos_token_id
            elif (
                    hasattr(seq2seq_model.config, "decoder")
                    and hasattr(seq2seq_model.config.decoder, "bos_token_id")
                    and seq2seq_model.config.decoder.bos_token_id is not None
            ):
                decoder_start_token_id = seq2seq_model.config.decoder.bos_token_id
            else:
                raise ValueError(
                    "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                )

        assert hasattr(seq2seq_model, "get_encoder"), "{} should have a 'get_encoder' function defined".format(seq2seq_model)
        assert callable(seq2seq_model.get_encoder), "{} should be a method".format(seq2seq_model.get_encoder)

        # get encoder and store encoder outputs
        encoder = seq2seq_model.get_encoder()
        encoder_outputs: ModelOutput = encoder(input_ids, position_ids=pos_ids, attention_mask=attention_mask, return_dict=True)

    # Expand input ids if num_beams > 1 or num_return_sequences > 1
    if num_return_sequences > 1 or num_beams > 1:
        input_ids_len = input_ids.shape[-1]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
        attention_mask = attention_mask.unsqueeze(1).expand(
            batch_size, effective_batch_mult * num_beams, input_ids_len
        )

        input_ids = input_ids.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        attention_mask = attention_mask.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    if seq2seq_model.config.is_encoder_decoder:
        # create empty decoder input_ids
        input_ids = torch.full(
            (effective_batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(seq2seq_model.parameters()).device,
        )
        cur_len = 1

        assert (
                batch_size == encoder_outputs.last_hidden_state.shape[0]
        ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
        )

        # expand encoder_outputs
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_batch_idxs
        )

        # save encoder_outputs in `model_kwargs`
        model_kwargs["encoder_outputs"] = encoder_outputs

    else:
        cur_len = input_ids.shape[-1]

    assert (
            cur_len < max_length
    ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

    output = _generate_no_beam_search(
        seq2seq_model,
        input_ids,
        cur_len=cur_len,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        bad_words_ids=bad_words_ids,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        batch_size=effective_batch_size,
        attention_mask=attention_mask,
        use_cache=use_cache,
        model_kwargs=model_kwargs,
    )

    return output


def _generate_no_beam_search(
    seq2seq_model: EncoderDecoderModel,
    input_ids,
    cur_len,
    max_length,
    min_length,
    do_sample,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    bad_words_ids,
    pad_token_id,
    eos_token_id,
    batch_size,
    attention_mask,
    use_cache,
    model_kwargs,
):
    """Generate sequences for each example without beam search (num_beams == 1).
    All returned sequence are generated independantly.
    """
    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(max_length)

    past = None
    while cur_len < max_length:
        model_inputs = seq2seq_model.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
        )

        outputs = seq2seq_model(**model_inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

        scores = seq2seq_model.postprocess_next_token_scores(
            scores=next_token_logits,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
        )

        # if model has past, then set the past variable to speed up decoding
        if "past_key_values" in outputs:
            past = outputs.past_key_values
        elif "mems" in outputs:
            past = outputs.mems

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        # add token and increase length by one
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        cur_len = cur_len + 1

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if seq2seq_model.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    return input_ids


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

















