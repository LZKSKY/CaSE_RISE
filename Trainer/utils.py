from config_n import default_config as config
import numpy as np
from Preprocess.utils_ import Tag
max_num = 1e8
min_num = 1e-8
K = config.K


def cal_cost(seq_a, seq_b, prob):
    # return matrix saving (cost, operations)
    # offset = 1   # 0 padding
    edit_mapping = config.edit2id
    cost_arr = 1 - prob + 1e-8

    def cal_cost(i, j, ops='d'):
        ind = edit_mapping[ops]
        return cost_arr[i, ind]

    n = len(seq_a)
    m = len(seq_b)
    # C = [[0 for j in range(0, m + 1)] for i in range(0, n + 1)]
    C = np.zeros([n + 1, m + 1])
    P = np.zeros([n + 1, m + 1, 4])
    # d = [[(0, '') for j in range(0, m + 1)] for i in range(0, n + 1)]  # [n+1, m+1, 2]
    for i in range(1, n + 1):
        C[i][0] = C[i - 1][0] + cal_cost(i, 0, 'D')
    for j in range(1, m + 1):
        C[0][j] = C[0][j - 1] + cal_cost(0, j - 1, 'I')
    '''
        c same     i insert     d delete        s sub
        operation of a change to b
    '''
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = seq_a[i - 1] == seq_b[j - 1]
            cost = [C[i - 1, j - 1] + cal_cost(i, j, 'K'),
                    C[i - 1, j] + cal_cost(i, j, 'D'),
                    C[i, j - 1] + cal_cost(i, j, 'I'),
                    C[i - 1, j - 1] + cal_cost(i, j, 'S')]
            cost = np.array(cost)
            norm_prob = (cost - max(cost)) / (min(cost) - max(cost))
            norm_prob_exp = np.exp(norm_prob * K)
            if same:
                norm_prob_exp[-1] = 0.
            else:
                norm_prob_exp[0] = 0.
            P[i][j] = norm_prob_exp / np.sum(norm_prob_exp)
            C[i][j] = np.sum(P[i][j] * cost)
    return C, P


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


def K_sampling(tag_arr, prob):
    # new_tag_arr = []
    l = len(tag_arr)
    p = np.random.random(size=l)
    rep = p <= prob[:l, config.edit2id['K']]
    for i, tag in enumerate(tag_arr):
        if rep[i]:
            tag_arr[i] = Tag('K', [])


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
    for index, tag in enumerate(action):
        act, seq = tag.ope, tag.seq
        if act == 'K':
            new_arr.append(seq_a[index])
        elif act == 'I':
            new_arr.extend([seq_a[index]] + seq)
        elif act == 'S':
            new_arr.extend(seq)
    # BOS + seq_a + EOS
    return new_arr


# def check_action(seq_a, seq_b, P):
#     ops_arr = sample_action_by_p(seq_a, seq_b, P)
#     seq_gen = apply_action(seq_a, ops_arr)
#     print('')
#
#
# if __name__ == '__main__':
#     seq_a = [1, 1, 2, 3, 4, 5, 6, 7, 7, 8]
#     seq_b = [1, 2, 3, 4, 5, 6, 7, 8]
#     prob = np.zeros([len(seq_a) + 1, 5])
#     prob[:, 1] = 1.
#     C, P = cal_cost(seq_a, seq_b, prob)
#     # action = sample_action(seq_a, seq_b, P)
#     check_action(seq_a, seq_b, P)
