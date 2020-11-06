
def pad(ori_arr, pad_value, desired_num):
    assert desired_num > 0
    result = ori_arr[:desired_num] + [pad_value] * (desired_num - len(ori_arr))
    assert len(result) == desired_num
    return result


def default_cost(i, j, d):
    return 1


def small_cost(i, j, d):
    return 0.5


def huge_cost(i, j, d):
    return 100000000


def edit_matrix(a, b,
                sub_cost=default_cost,
                ins_cost=default_cost,
                del_cost=default_cost,
                trans_cost=default_cost):
    n = len(a)
    m = len(b)
    d = [[(0, '') for j in range(0, m+1)] for i in range(0, n+1)]   # [n+1, m+1, 2]
    for i in range(1, n + 1):
        d[i][0] = (d[i - 1][0][0] + del_cost(i, 0, d), d[i-1][0][1] + 'd')
    for j in range(1, m + 1):
        d[0][j] = (d[0][j - 1][0] + ins_cost(0, j, d), d[0][j-1][1] + 'i')
    '''
        c same     i insert     d delete        s sub
        operation of a change to b
    '''
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:    # equals, no cost
                d[i][j] = (d[i - 1][j - 1][0], d[i - 1][j - 1][1] + 'c')
            else:
                '''
                    del a[i]
                    insert b[j] after a[i]
                    replace a[i] with b[j], or reversed
                '''
                d[i][j] = min(
                    (d[i - 1][j][0] + del_cost(i, j, d), d[i - 1][j][1] + 'd'),
                    (d[i][j - 1][0] + ins_cost(i, j, d), d[i][j - 1][1] + 'i'),
                    (d[i - 1][j - 1][0] + sub_cost(i, j, d), d[i - 1][j - 1][1] + 's')
                )
                can_transpose = (
                    i > 2 and
                    j > 2 and
                    a[i - 1] == b[j - 2] and
                    a[i - 2] == b[j - 1]
                )
                if can_transpose:
                    d[i][j] = min(
                        d[i][j],
                        (d[i - 2][j - 2][0] + trans_cost(i, j, d), d[i - 2][j - 2][1] + 't')
                    )
    return d


def edit_diff(a, b, d):
    script = d[-1][-1][1]
    diff = []
    i = j = 0
    for k in range(0, len(script)):
        if script[k] == 'c':
            diff.append(('c', a[i], b[j]))
            i += 1
            j += 1
        elif script[k] == 's':
            diff.append(('s', a[i], b[j]))
            i += 1
            j += 1
        elif script[k] == 'd':
            diff.append(('d', a[i]))
            i += 1
        elif script[k] == 'i':
            diff.append(('i', b[j]))
            j += 1
        else:
            raise Exception('Unsupported operation')
    return diff


def extend_supervision(output_query, query_len, pad_id):
    output_query_r_edit = []  # corresponding origin sentence
    output_query_edit = []  # corresponding gold sentence
    distance_supervision = []
    cache = []
    index = 1  # CLS_WORD
    for j, e in enumerate(output_query):
        if e[0] == 'i':
            if index < query_len:
                cache.append(index)
            index += 1
            output_query_r_edit.append(e)  # tell origin sentence, u need to insert
        elif e[0] == 'd':
            distance_supervision.append(cache)  # tell gold sentence, this need to anaphora resolution
            output_query_edit.append(e)  # tell gold sentence, this need to delete
        elif e[0] == 'c':
            index += 1
            cache = []
            distance_supervision.append([])
            output_query_r_edit.append(e)
            output_query_edit.append(e)
        elif e[0] == 's':
            output_query_r_edit.append(e)
            output_query_edit.append(e)
        else:
            print('Unexpected')
    distance_supervision = [pad(dist, pad_id, 4) for dist in distance_supervision]

    return output_query_r_edit, output_query_edit, distance_supervision

