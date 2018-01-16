import numpy as np


def load_prob_stats(stats_fn, cnt):
    fp = open(stats_fn, 'r')

    max_label_vec = np.zeros(cnt, dtype=np.int32)
    prob_avg_vec = np.zeros(cnt, dtype=np.float32)

    line_cnt = 0
    for line in fp:
        line_cnt += 1
        if line_cnt == 1:
            continue
        if line_cnt > cnt:
            break

        spl = line.split()
        prob_avg_vec[line_cnt - 1] = float(spl[2])
        max_label_vec[line_cnt - 1] = int(spl[1])

    fp.close()

    return prob_avg_vec, max_label_vec
