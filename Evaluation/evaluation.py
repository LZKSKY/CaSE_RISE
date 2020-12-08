from config_n import Config
from six.moves import map

from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
config = Config()
quac_dataset_path, output_path = config.quac_dataset_path, config.output_path


def gen_evaluation(hyp_list, ref_list):

    def _strip(s):
        return s.strip()

    ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hyps)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                print("%s: %0.6f" % (m, sc))
                ret_scores[m] = sc
        else:
            print("%s: %0.6f" % (method, score))
            ret_scores[method] = score
    del scorers


def compute_score(file_name, turn_arr: list):
    import torch
    from tqdm import tqdm
    all_arr = torch.load(file_name)
    refs, hyps1, hyps2 = [], [], []
    # rank_list = []
    hyps = [[] for i in range(len(turn_arr))]
    for obj in tqdm(all_arr):
        refs.append(obj['output_query'])
        for index, t in enumerate(turn_arr):
            hyps[index].append(obj['gen_query'][index])
        # hyps2.append(obj['input_query'])
    print('generate score is ')
    for index, t in enumerate(turn_arr):
        print(f'score for turn {t} is: ')
        gen_evaluation(hyps[index], (refs,))


def compute_score_mle(file_name):
    import torch
    from tqdm import tqdm
    all_arr = torch.load(file_name)
    refs, hyps1, hyps2 = [], [], []
    # rank_list = []
    hyps = []
    for obj in tqdm(all_arr):
        # print(obj)
        refs.append(obj['output_query'])
        hyps.append(obj['gen_query'])
        # hyps2.append(obj['input_query'])
    print('generate score is ')
    gen_evaluation(hyps, (refs,))
