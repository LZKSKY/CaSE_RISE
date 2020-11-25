def pad(ori_arr, pad_value, desired_num, padding_mode='r'):
    assert desired_num > 0
    if padding_mode == 'r':
        result = ori_arr[:desired_num] + [pad_value] * (desired_num - len(ori_arr))
    elif padding_mode == 'l':
        result = [pad_value] * (desired_num - len(ori_arr)) + ori_arr[-desired_num:]
    else:
        result = ori_arr[:desired_num]
    assert len(result) == desired_num
    return result




