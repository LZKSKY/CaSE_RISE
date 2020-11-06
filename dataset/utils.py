def pad(ori_arr, pad_value, desired_num):
    assert desired_num > 0
    result = ori_arr[:desired_num] + [pad_value] * (desired_num - len(ori_arr))
    assert len(result) == desired_num
    return result




