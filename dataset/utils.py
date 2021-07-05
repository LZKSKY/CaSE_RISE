from Preprocess.utils_ import Tag


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


def merge_tag(tag_arr):
    desired_len = len(tag_arr)
    index = 0
    # change  "DSDKK" -> "SSSKK"
    while index < desired_len:
        if tag_arr[index].ope == 'S':
            w_arr = tag_arr[index].seq
            p_l = index - 1
            while p_l >= 0 and tag_arr[p_l].ope == 'D':
                p_l -= 1
            p_r = index + 1
            while p_r < desired_len and tag_arr[p_r].ope in 'DS':
                if tag_arr[p_r].ope == 'S':
                    w_arr.extend(tag_arr[p_r].seq)
                p_r += 1
            for temp_index in range(p_l + 1, p_r):
                tag_arr[temp_index] = Tag('S', w_arr)
            index = p_r
        else:
            index += 1

    return tag_arr





