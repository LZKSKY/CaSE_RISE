from config_n import default_config as config
import numpy as np
from Preprocess.utils_ import Tag
max_num = 1e16
min_num = 1e-16


def cal_prob_matrix(seq_a, seq_b, model_prob, trunc=0., part_credit=True):
    # return matrix saving (cost, operations)
    # offset = 1   # 0 padding
    edit_mapping = config.edit2id
    # cost_arr = prob

    def cal_score(i, j, ops='D'):
        ind = edit_mapping[ops]
        return model_prob[i, ind]

    n = len(seq_a)
    m = len(seq_b)
    # C = [[0 for j in range(0, m + 1)] for i in range(0, n + 1)]
    M = np.zeros([n + 1, m + 1])        # log_p
    P = np.zeros([n + 1, m + 1, 4])
    # d = [[(0, '') for j in range(0, m + 1)] for i in range(0, n + 1)]  # [n+1, m+1, 2]
    if not part_credit:
        trans_func = np.exp
        inverse_func = np.log
        M[0, 0] = cal_score(0, 0, 'K') + 1e-8
    else:
        inverse_func = trans_func = lambda x: x
        M[0, 0] = cal_score(0, 0, 'K') + 1e-8

    for i in range(1, n + 1):
        d_prob = cal_score(i, 0, 'D')
        M[i][0] = M[i - 1][0] + d_prob
        P[i][0] = [0, d_prob, 0, 0]
    for j in range(1, m + 1):
        i_prob = cal_score(0, j - 1, 'I')
        M[0][j] = M[0][j - 1] + i_prob
        P[0][j] = [0, 0, i_prob, 0]
    '''
        c same     i insert     d delete        s sub
        operation of a change to b
    '''
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # if i == 3 and j == 3:
            #     print()
            same = seq_a[i - 1] == seq_b[j - 1]
            score = [trans_func(M[i - 1, j - 1] + cal_score(i, j, 'K')),
                     trans_func(M[i - 1, j] + cal_score(i, j, 'D')),
                     trans_func(M[i, j - 1] + cal_score(i, j, 'I')),
                     trans_func(M[i - 1, j - 1] + cal_score(i, j, 'S'))]
            score = np.array(score)     # saves prob product
            score[score < min_num] = min_num
            if same:
                score[1] = score[2] = score[3] = 0.  # [D, S, I]
            else:
                score[0] = 0.  # [K]
            # ============= modify, M need to product P =========================
            if np.sum(score) == 0:
                p = np.zeros(score)
            else:
                p = score / np.sum(score)
                if trunc > 0:
                    p[p < trunc] = 0.
                    p = score / np.sum(score)
            P[i][j] = p
            M[i][j] = inverse_func(np.sum(score * P[i][j]) + min_num)
            # if M[i][j] < 0:
            #     print('')
            # M[i][j] = np.log(np.sum(score) + min_num)
    return M, P


def sample_action_by_p(seq_a, seq_b, P):
    # in: m, n, m + 1 * n + 1
    # return m + 2
    # note that, we return a seq of n + 1 operation of [BOS] + [seq_a] + [EOS]
    n = len(seq_a)
    m = len(seq_b)
    i = n
    j = m
    ops_arr = []
    record_seq = []
    sample_low_prob = False
    while i > 0 and j > 0:
        # if not record_seq:  # if some word is insert, do not Delete or Substitute
        #     P[i][j][1] = P[i][j][3] = 0.
        #     P[i][j] = P[i][j] / np.sum(P[i][j])
        ops = np.random.choice(['K', 'D', 'I', 'S'], p=P[i][j])
        seq = seq_b[j - 1] if ops in ['S', 'I'] else ''
        if seq:
            record_seq.insert(0, seq)

        if ops in ['K', 'D', 'S']:
            if not record_seq:  # ['K', 'D']
                ops_arr.append([i, ops, []])
            else:   # 'K' + ['I'] * n   or 'S'
                if ops == 'S':
                    ops_arr.append([i, ops, record_seq])
                    record_seq = []
                elif ops == 'D':
                    ops_arr.append([i, 'D', []])
                elif ops == 'K':
                    ops_arr.append([i, 'I', record_seq])
                    record_seq = []
                # TODO:
                # ops = 'S' if ops == 'S' else 'I'
                # ops_arr.append([i, ops, record_seq])
                # record_seq = []

        if ops in ['K', 'D', 'S']:
            i -= 1
        if ops in ['K', 'I', 'S']:
            j -= 1
    while i > 0:
        if P[i][j][config.edit2id['D']] <= 0.2:
            sample_low_prob = True
        ops_arr.append([i, 'D', []])
        i -= 1
    while j > 0:
        seq = seq_b[j - 1]
        if seq:
            record_seq.insert(0, seq)
        j -= 1
    if record_seq:
        if P[i][j][config.edit2id['I']] <= 0.2:
            sample_low_prob = True
        ops_arr.append([i, 'I', record_seq])
        # ops_arr.append([0, 'I', record_seq])
    else:
        ops_arr.append([0, 'K', record_seq])

    tag_arr = [Tag(ops, seq) for index, ops, seq in list(reversed(ops_arr))]
    tag_arr += [Tag('K', [])]
    assert len(tag_arr) == len(seq_a) + 2
    return tag_arr, sample_low_prob


def greedy_action_by_p(seq_a, seq_b, P):
    id2edit = config.id2edit
    # in: m, n, m + 1 * n + 1
    # return m + 2
    # note that, we return a seq of n + 1 operation of [BOS] + [seq_a] + [EOS]
    n = len(seq_a)
    m = len(seq_b)
    i = n
    j = m
    ops_arr = []
    record_seq = []
    while i > 0 and j > 0:
        ops_index = np.argmax(P[i][j])
        ops = id2edit[ops_index + 1]
        seq = seq_b[j - 1] if ops in ['S', 'I'] else ''
        if seq:
            record_seq.insert(0, seq)

        if ops in ['K', 'D', 'S']:
            if not record_seq:  # ['K', 'D']
                ops_arr.append([i, ops, []])
            else:   # 'K' + ['I'] * n   or 'S'
                if ops == 'S':
                    ops_arr.append([i, ops, record_seq])
                    record_seq = []
                elif ops == 'D':
                    ops_arr.append([i, 'D', []])
                elif ops == 'K':
                    ops_arr.append([i, 'I', record_seq])
                    record_seq = []
                # TODO:
                # ops = 'S' if ops == 'S' else 'I'
                # ops_arr.append([i, ops, record_seq])
                # record_seq = []

        if ops in ['K', 'D', 'S']:
            i -= 1
        if ops in ['K', 'I', 'S']:
            j -= 1
    while i > 0:
        ops_arr.append([i, 'D', []])
        i -= 1
    while j > 0:
        seq = seq_b[j - 1]
        if seq:
            record_seq.insert(0, seq)
        j -= 1
    if record_seq:
        ops_arr.append([0, 'I', record_seq])
    else:
        ops_arr.append([0, 'K', record_seq])

    tag_arr = [Tag(ops, seq) for index, ops, seq in list(reversed(ops_arr))]
    tag_arr += [Tag('K', [])]
    assert len(tag_arr) == len(seq_a) + 2
    return tag_arr


def sampling(prob, L, sampling_strategy='softmax', eps=0.2, tag=False):
    prob = np.array(prob)
    assert len(prob.shape) == 2
    if sampling_strategy == 'greedy':
        all_len = prob.shape[1] - 2     # 1 for padding, 1 for max
        arg_index = prob.argmax(axis=1)
        prob = np.ones_like(prob) * (eps / all_len)
        prob[:, 0] = 0.
        prob[np.arange(prob.shape[0]), arg_index] = 1 - eps
    arr = []
    for i in range(L):
        v = np.random.choice(len(prob[i]), p=prob[i])
        assert v > 0
        if tag:
            v = Tag(config.id2edit[v], [])
        arr.append(v)
    return arr


def apply_action(seq_a, action):
    new_arr = []
    # seq_a = ['BOS'] + seq_a + ['EOS]
    assert len(action) == len(seq_a)
    s_pass = False
    for index, tag in enumerate(action):
        act, seq = tag.ope, tag.seq
        if act != 'S':
            s_pass = False
        if act == 'K':
            new_arr.append(seq_a[index])
        elif act == 'I':
            new_arr.extend([seq_a[index]] + seq)
        elif not s_pass and act == 'S':
            new_arr.extend(seq)
            s_pass = True
    # BOS + seq_a + EOS
    return new_arr


def re_assign_K(seq_a, seq_b, action):
    if action[0].ope not in ['K', 'I']:
        action[0] = Tag('K', [])
    action[-1] = Tag('K', [])
    for index, k in enumerate(seq_a):
        if k in seq_b and action[index + 1].ope not in ['K', 'I']:
            action[index + 1] = Tag('K', [])
        if k not in seq_b and action[index + 1].ope not in ['S', 'D']:
            action[index + 1] = Tag('D', [])
    return action

