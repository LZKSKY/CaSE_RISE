from copy import deepcopy
from typing import List
from config_n import default_config
D_id = default_config.edit2id['D']
S_id = default_config.edit2id['S']


def pad(ori_arr, pad_value, desired_num, padding_mode='r'):
    assert desired_num > 0
    if padding_mode == 'r':
        result = ori_arr[:desired_num] + [pad_value] * (desired_num - len(ori_arr))
    elif padding_mode == 'l':
        result = [pad_value] * (desired_num - len(ori_arr)) + ori_arr[:desired_num]
    else:
        result = ori_arr[:desired_num]
    assert len(result) == desired_num
    return result


def merge_path(trajectory: List):
    source_len = len(trajectory)
    index = 0
    while index < source_len:
        if trajectory[index] in [S_id]:
            p_l = index - 1
            while p_l >= 0 and trajectory[p_l] in [D_id]:
                trajectory[p_l] = S_id
                p_l -= 1
            p_r = index + 1
            while p_r < source_len and trajectory[p_r] in [D_id, S_id]:
                trajectory[p_r] = S_id
                p_r += 1
            index = p_r
        else:
            index += 1
    return trajectory





