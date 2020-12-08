from copy import deepcopy
from typing import List

DEFAULT_COST = 1.
SMALL_COST = DEFAULT_COST * 0.5
HUGE_COST = DEFAULT_COST * 1e-8


class Tag(object):
    def __init__(self, ope, para):
        self.ope = ope
        self.seq: List = para


class TagSeq:
    def __init__(self, sub_cost=DEFAULT_COST, ins_cost=DEFAULT_COST, del_cost=DEFAULT_COST):
        self.set_cost(sub_cost, ins_cost, del_cost)

    def get_func(self, num):
        return lambda *args: num

    def set_cost(self, sub_cost=DEFAULT_COST, ins_cost=DEFAULT_COST, del_cost=DEFAULT_COST):
        self.sub_cost = sub_cost if hasattr(sub_cost, '__call__') else self.get_func(sub_cost)
        self.ins_cost = ins_cost if hasattr(ins_cost, '__call__') else self.get_func(ins_cost)
        self.del_cost = del_cost if hasattr(del_cost, '__call__') else self.get_func(del_cost)

    def get_edit_matrix(self, seq_a, seq_b):
        # return matrix saving (cost, operations)
        n = len(seq_a)
        m = len(seq_b)
        d = [[(0, '') for j in range(0, m + 1)] for i in range(0, n + 1)]  # [n+1, m+1, 2]
        for i in range(1, n + 1):
            d[i][0] = (d[i - 1][0][0] + self.del_cost(i, 0, d), d[i - 1][0][1] + 'd')
        for j in range(1, m + 1):
            d[0][j] = (d[0][j - 1][0] + self.ins_cost(0, j, d), d[0][j - 1][1] + 'i')
        '''
            c same     i insert     d delete        s sub
            operation of a change to b
        '''
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if seq_a[i - 1] == seq_b[j - 1]:  # equals, no cost
                    d[i][j] = (d[i - 1][j - 1][0], d[i - 1][j - 1][1] + 'c')
                else:
                    '''
                        del a[i]
                        insert b[j] after a[i]
                        replace a[i] with b[j], or reversed
                    '''
                    d[i][j] = min(
                        (d[i - 1][j][0] + self.del_cost(i, j, d), d[i - 1][j][1] + 'd'),
                        (d[i][j - 1][0] + self.ins_cost(i, j, d), d[i][j - 1][1] + 'i'),
                        (d[i - 1][j - 1][0] + self.sub_cost(i, j, d), d[i - 1][j - 1][1] + 's')
                    )
        return d

    def sample_path(self, matrix, seq_a, seq_b, greedy_prob=0.8):
        trajectory = []
        index_i, index_j = len(seq_a), len(seq_b)
        ope_arr = []

        while index_i > 0 and index_j > 0:
            if matrix[index_i][index_j][1][-1] == 's':
                ope_arr.insert(0, 'S')
            else:
                prob = (matrix[index_i - 1][index_j][0], matrix[index_i][index_j - 1][0],
                        matrix[index_i - 1][index_j - 1][0])

        return trajectory

    def get_path(self, matrix, seq_a, seq_b):
        trajectory = []
        source_index = 0
        target_index = 0
        for ope in matrix[-1][-1][1]:
            if ope == 'c':
                trajectory.append(Tag('K', []))
                source_index += 1
                target_index += 1
            elif ope == 's':
                trajectory.append(Tag('S', [seq_b[target_index]]))
                source_index += 1
                target_index += 1
            elif ope == 'd':
                trajectory.append(Tag('D', []))
                source_index += 1
            elif ope == 'i':
                trajectory.append(Tag('I', [seq_b[target_index]]))
                target_index += 1
            else:
                raise Exception('Unsupported operation')
        return trajectory

    def merge_path(self, trajectory: List[Tag], desired_len: int):
        pre_state = 'K'

        # change  "DSDKK" -> "SSSKK"
        source_len = len(trajectory)
        index = 0
        while index < source_len:
            if trajectory[index].ope == 'S':
                w_arr = deepcopy(trajectory[index].seq)
                p_l = index - 1
                while p_l >= 0 and trajectory[p_l].ope == 'D':
                    p_l -= 1
                p_r = index + 1
                while p_r < source_len and trajectory[p_r].ope in 'DS':
                    if trajectory[p_r].ope == 'S':
                        w_arr.extend(trajectory[p_r].seq)
                    p_r += 1
                for temp_index in range(p_l + 1, p_r):
                    trajectory[temp_index] = Tag('S', w_arr)
                index = p_r
            else:
                index += 1

        # change "KKIIIKK" to "KIKK"
        new_trajectory: List[Tag] = []
        temp_cache = []
        trajectory = [Tag('K', [])] + trajectory
        for tag in trajectory:
            if tag.ope == 'I':
                temp_cache.extend(tag.seq)
            elif temp_cache:
                if len(new_trajectory) >= 2 and new_trajectory[-2].ope == 'S' and new_trajectory[-1].ope == 'S':
                    new_trajectory[-1].seq = deepcopy(temp_cache)
                else:
                    new_trajectory[-1].seq.extend(temp_cache)
                new_trajectory[-1].ope = 'I'
                temp_cache = []
                new_trajectory.append(deepcopy(tag))
            else:
                new_trajectory.append(deepcopy(tag))
        if temp_cache:
            if len(new_trajectory) >= 2 and new_trajectory[-2].ope == 'S' and new_trajectory[-1].ope == 'S':
                new_trajectory[-1].seq = deepcopy(temp_cache)
            else:
                new_trajectory[-1].seq.extend(temp_cache)
            new_trajectory[-1].ope = 'I'
            # temp_cache = []
        assert len(new_trajectory) == desired_len
        return new_trajectory

    def return_length(self, trajectory: List[Tag]):
        length = 0
        for tag in trajectory:
            if tag.ope == 'S':
                length += 1
            elif tag.ope == 'D':
                length += 1
            elif tag.ope == 'I':
                length += len(tag.seq)
        return length

    def get_label(self, input_seq, output_seq, return_length=False):
        # derived label from matrix
        matrix = self.get_edit_matrix(input_seq, output_seq)    # [m + 1, n + 1]
        trajectory = self.get_path(matrix, input_seq, output_seq)   # [traj]
        new_trajectory = self.merge_path(trajectory, len(input_seq) + 1)    # [m + 1]
        # new_trajectory = trajectory
        # TODO: check length and design standard
        if return_length:
            return new_trajectory, self.return_length(trajectory)
        else:
            return new_trajectory


def obtain_gen_seq(tag_arr: List[Tag], input_query):
    if len(tag_arr) == len(input_query) + 1:
        tag_arr = tag_arr[:-1]
    input_len = len(tag_arr)
    i = 0
    gen_seq_arr = []
    while i < input_len:
        if tag_arr[i].ope == 'I':
            gen_seq_arr.append(([input_query[i], input_query[i + 1]], tag_arr[i].seq))
            i += 1
        elif tag_arr[i].ope == 'S':
            gen_seq_arr.append(([input_query[i]], tag_arr[i].seq))
            i += 1
            while i < input_len and tag_arr[i].ope == 'S':
                gen_seq_arr[-1][0].append(input_query[i])
                i += 1
        else:
            i += 1
    return gen_seq_arr







