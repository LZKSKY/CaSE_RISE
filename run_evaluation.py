from nlgeval import compute_metrics
from config_n import Config
import argparse
config = Config()
quac_dataset_path, output_path = config.quac_dataset_path, config.output_path


def gen_evaluation(hyp_list, ref_list):
    import six
    from six.moves import map

    from nlgeval.pycocoevalcap.bleu.bleu import Bleu
    from nlgeval.pycocoevalcap.cider.cider import Cider
    from nlgeval.pycocoevalcap.meteor.meteor import Meteor
    from nlgeval.pycocoevalcap.rouge.rouge import Rouge
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


def compute_score(file_name):
    import torch
    from tqdm import tqdm
    all_arr = torch.load(file_name)
    refs, hyps1, hyps2 = [], [], []
    # rank_list = []
    for obj in tqdm(all_arr):
        refs.append(obj['output_query'])
        hyps1.append(obj['gen_query'])
        hyps2.append(obj['input_query'])
    print('generate score is ')
    gen_evaluation(hyps1, (refs,))
    print('input query score is')
    gen_evaluation(hyps2, (refs,))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_epoch", type=int, default=24)
    parser.add_argument("--model_name", type=str, default="bert_mld")
    args = parser.parse_args()
    final_path = f'{output_path}{args.model_name}-{args.load_epoch}_gen_out.pkl'
    compute_score(final_path)
    # compute_score('../QuacN/Output/bert2bertEdit-all-6_eval_out.pkl')

