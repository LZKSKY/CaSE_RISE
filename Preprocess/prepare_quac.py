import json
import os
from random import shuffle
from tqdm import tqdm


def save_json(obj, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)


def gen_desired_quac(train_arr, dev_arr, test_arr, ref_query_arr):
    '''
           quac.query              #query_id	#query
           quac.answer             #context_id	#query_id	#passage_id	#answer	#answer_start
           quac.origin.answer      #context_id	#query_id	#passage_id	#answer	#answer_start
           quac.auxiliary.answer   #query_id	#followup	#yesno
           quac.auxiliary.passage     #passage_id	#title	#section_title
           quac,passage            #passage_id	#passage
           ..quac.qrel               #query_id #0 #passage_id #relevance
           ..quac.reformulated_query    #query_id	#reformulated_query
           ..quac.split              #query_id	#split
           ..quac.case_s             #query_id	#query_candidate
       '''
    concat_arr = train_arr + dev_arr + test_arr
    quac = {'query': [], 'answer': [], 'origin.answer': [], 'auxiliary.answer': [], 'auxiliary.passage': [],
            'passage': [], 'split': [], 'reformulated_query': []}
    pagid2qas = dict([(k['paragraphs'][0]['id'], k['paragraphs'][0]['qas']) for k in concat_arr])
    for i, tup in tqdm(enumerate(concat_arr)):
        if i < len(train_arr):
            mod = 'train'
        elif i < len(train_arr) + len(dev_arr):
            mod = 'dev'
        else:
            mod = 'test'
        pasg = tup['paragraphs']
        assert len(pasg) == 1
        pasg = pasg[0]
        pasg_id, b_id, c_id = pasg['id'], pasg['id'] + '_b', pasg['id'] + '_c'
        quac['passage'].extend([[b_id, tup['background']], [c_id, pasg['context']]])
        quac['auxiliary.passage'].extend([[b_id, tup['title'], tup['section_title']],
                                          [c_id, tup['title'], tup['section_title']]])
        for j, qas in enumerate(pasg['qas']):
            q_id = qas['id']
            quac['split'].append([q_id, mod])
            quac['query'].append([q_id, qas['question']])
            context_arr, cur_query, pasg_id_arr = [k['id'] for k in pasg['qas'][:j]], qas['id'], [b_id, c_id]
            for k, ans in enumerate(qas['answers']):
                quac['answer'].append([';'.join(context_arr),
                                       cur_query,
                                       ';'.join(pasg_id_arr),
                                       ans['text'], ans['answer_start']])
            quac['origin.answer'].append([';'.join(context_arr),
                                          q_id,
                                          ';'.join(pasg_id_arr),
                                          qas['orig_answer']['text'], qas['orig_answer']['answer_start']])
            quac['auxiliary.answer'].append([q_id, qas['followup'], qas['yesno']])

    for i, d in enumerate(ref_query_arr):
        pag_id = d['QuAC_dialog_id']
        q_no = d['Question_no'] - 1
        qas = pagid2qas[pag_id][q_no]
        ori_q = d['Question']
        tar_q = d['Rewrite']
        while True:
            if ori_q == qas['question']:
                break
            else:
                q_no += 1
                if q_no >= len(pagid2qas[pag_id]):
                    raise IndexError
                qas = pagid2qas[pag_id][q_no]
        assert ori_q == qas['question']
        quac['reformulated_query'].append([qas['id'], tar_q])

    return quac


def quac_split_data(path='../QuacN/', out_path='../QuacN/processed/', canard_path='../CANARD_Release/'):
    ori_train = path + 'train_v0.2.json'
    ori_dev = path + 'val_v0.2.json'
    can_train = load_json(canard_path + 'train.json')
    can_dev = load_json(canard_path + 'dev.json')
    can_test = load_json(canard_path + 'test.json')
    if os.path.exists(path + 'train.json'):
        train_arr = load_json(path + 'train.json')
        dev_arr = load_json(path + 'dev.json')
        test_arr = load_json(path + 'test.json')
    else:
        dev_dialog_id = set([k['QuAC_dialog_id'] for k in can_dev])
        train_dialog_id = set([k['QuAC_dialog_id'] for k in can_train])

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        ori_train_arr = load_json(ori_train)['data']
        ori_dev_arr = load_json(ori_dev)['data']
        test_arr = ori_dev_arr

        train_arr, dev_arr = [], []
        for pasg in ori_train_arr:
            cur_dialogg_id = pasg['paragraphs'][0]['id']
            if cur_dialogg_id in dev_dialog_id:
                assert cur_dialogg_id not in train_dialog_id
                dev_arr.append(pasg)
            else:
                train_arr.append(pasg)

        save_json(train_arr, path + 'train.json')
        save_json(dev_arr, path + 'dev.json')
        save_json(test_arr, path + 'test.json')

    '''
        quac.query              #query_id	#query
        quac.answer             #context_id	#query_id	#passage_id	#answer	#answer_start
        quac.origin.answer      #context_id	#query_id	#passage_id	#answer	#answer_start
        quac.auxiliary.answer   #query_id	#followup	#yesno
        quac.auxiliary.passage     #passage_id	#title	#section_title
        quac,passage            #passage_id	#passage
        quac.qrel               #query_id #0 #passage_id #relevance
        quac.reformulated_query    #query_id	#reformulated_query
        quac.split              #query_id	#split
        quac.case_s             #query_id	#query_candidate
    '''

    quac = gen_desired_quac(train_arr, dev_arr, test_arr, can_train + can_dev + can_test)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    k_arr = {'query': '#query_id	#query',
             'answer': '#context_id	#query_id	#passage_id	#answer	#answer_start',
             'origin.answer': '#context_id	#query_id	#passage_id	#answer	#answer_start',
             'auxiliary.answer': '#query_id	#followup	#yesno',
             'auxiliary.passage': '#passage_id	#title	#section_title',
             'passage': '#passage_id	#passage',
             'split': '#query_id	#split',
             'reformulated_query': '#query_id	#reformulated_query'}
    for k, v in quac.items():
        f_name = out_path + 'quac.' + k
        with open(f_name, 'w', encoding='utf-8') as f:
            f.write(k_arr[k] + '\n')
            for tup in v:
                f.write('\t'.join([str(t) for t in tup]) + '\n')


if __name__ == '__main__':
    quac_split_data()
