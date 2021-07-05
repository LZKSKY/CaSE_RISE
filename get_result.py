import json
import numpy as np


def load_json(f_name):
    with open(f_name, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(f_name, obj, indent=None):
    with open(f_name, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def compare_arr(arr1, arr2):
    if len(arr1) != len(arr2):
        return False
    for index in range(len(arr1)):
        if arr1[index] != arr2[index]:
            return False
    return True


def check(result_file_1, result_file_2, eval_key='ROUGE_L'):
    result_1 = load_json(result_file_1)
    result_2 = load_json(result_file_2)
    arr = []
    for index, result in enumerate(result_1):
        k = list(result.keys())[0]
        v = result[k]
        arr.append([index, k, v, v[eval_key]])
    arr = sorted(arr, key=lambda x: x[-1], reverse=True)
    need_k = arr[0][1]
    need_key_arr = need_k.split('-')
    need_key_arr.remove('dev')
    for index, result in enumerate(result_2):
        k = list(result.keys())[0]
        cur_key_arr = k.split('-')
        if 'test' in cur_key_arr:
            cur_key_arr.remove('test')
        if 'CAsT_test' in cur_key_arr:
            cur_key_arr.remove('CAsT_test')
        if compare_arr(need_key_arr, cur_key_arr):
            print(need_k, k)
            for rk, rv in result[k].items():
                print(rk, np.round(arr[0][2][rk], 3), np.round(rv, 3))
        # v = result[k]


def log_result(path='../QuacN/Result/', method='RISE3', reward_strategy='R3', seed=None):

    f_path = path + f'{method}-'
    if reward_strategy is not None and reward_strategy:
        f_path += f'{reward_strategy}-'
    if seed is not None:
        f_path += f'{seed}-'
    print(f'==========================={f_path[len(path):]}=============================')
    f_dev_path = f_path + f'dev-result.json'
    f_test_path = f_path + f'test-result.json'
    f_test_path_cast = f_path + f'CAsT_test-result.json'
    try:
        check(f_dev_path, f_test_path_cast, eval_key='Bleu_4')
    except FileNotFoundError:
        return


if __name__ == '__main__':
    path = '../QuacN/Result/'
    reward_strategy = 'R2'
    log_result(path=path, method='RISE-L', reward_strategy=reward_strategy, seed=235)
    log_result(path=path, method='greedy', reward_strategy=reward_strategy, seed=235)
    log_result(path=path, method='softmax', reward_strategy=reward_strategy, seed=235)
    log_result(path=path, method='GPT2', reward_strategy='')
    log_result(path=path, method='BertPlus2', reward_strategy='')
    log_result(path=path, method='BertPlus', reward_strategy='')


















