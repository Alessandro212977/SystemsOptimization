from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from libraries.algorithms import *
from libraries.dataloader import DataLoader


def getTimetablePlot(TT, sigma, xmax=None, group_tt=False):
    # plt.rcParams["figure.figsize"] = [16, 7]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()

    # X-axis
    if not xmax:
        xmax = lcm(*[obj.period for obj in TT])
    ax.set_xlim(0 - 0.01 * xmax, xmax + 0.01 * xmax)
    # plt.xticks(np.arange(0, xmax+xmax//(xmax/1000), xmax//(xmax/1000)))
    ax.set_xlabel("Duration")

    # Y-axis
    y_ticklables = [obj.name for obj in TT]
    if group_tt:
        num_tt = sum([1 if "tt" in task.name.lower() else 0 for task in TT])
        y_ticklables = ["TT tasks"] + [obj.name for obj in TT[num_tt:]]

    plt.yticks(np.arange(0, len(y_ticklables), 1.0))
    ax.set_yticklabels(y_ticklables, va="bottom")
    ax.set_ylim(0, len(y_ticklables))
    ax.set_ylabel("Tasks")

    # data
    if group_tt:
        for i, val in enumerate(sigma):
            if val != "idle":
                sigma[i] = val if val >= num_tt else num_tt - 1
                sigma[i] = sigma[i] - num_tt + 1

    res = defaultdict(list)
    current_val = sigma[0]
    starting_idx = 0
    length = 1
    for i, val in enumerate(sigma[1:]):
        if val == current_val:
            length += 1
        else:
            res[current_val].append((starting_idx, length))
            length = 1
            starting_idx = i + 1
            current_val = val
    res[current_val].append((starting_idx, length))
    cmap = plt.cm.get_cmap("hsv", len(res))
    for key, value in res.items():
        if key == "idle" or key == "Idle":
            continue
        ax.broken_barh(value, (key, 1), facecolors=cmap(key))

    ax.set_title("Task Scheduling")
    ax.grid(True)

    return plt


if __name__ == "__main__":
    path = "./test_cases/taskset_small.csv"
    dl = DataLoader(path)
    TT, ET = dl.loadFile()
    ps = PollingServer("ps", duration=1000, period=2000, deadline=1000, tasks=ET, separation=0)
    schedulable, sigma, WCRT, __ = EDF(TT + [ps])
    # print(sigma, WCRT)
    getTimetablePlot(TT + [ps], sigma, group_tt=False)
    # plotSimulatedAnealing()
