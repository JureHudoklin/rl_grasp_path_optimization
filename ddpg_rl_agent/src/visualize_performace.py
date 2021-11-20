#! /usr/bin/python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

if __name__ == '__main__':
    # Load data
    run_id = "20211120-230404"
    data_dir = os.path.join(
        '/home/jure/reinforcement_ws/src/ddpg_rl_agent/src/checkpoints/', run_id)

    my_file = np.load(os.path.join(data_dir, '90.npz.npy'))
    running_average = running_mean(my_file, 10)
    plt.plot(running_average)
    plt.show()
