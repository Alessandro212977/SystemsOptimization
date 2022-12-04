from math import ceil, lcm, floor

import libraries.dataloader as dataloader
from libraries.tasks import PollingServer


def EDF(tasks):
    schedulable = True
    T = lcm(*[obj.period for obj in tasks])
    t = 0
    timetable = ["idle"] * T
    releases = [0] * len(tasks)  # release time
    wcrt = [-1] * len(tasks)  # -1 = deadline missed
    durations = [obj.duration for obj in tasks]  # durations
    deadlines = [obj.deadline for obj in tasks]  # deadlines

    penalty = [0] * len(tasks)

    while t < T:
        for i, task in enumerate(tasks):
            if durations[i] > 0 and deadlines[i] <= t:
                wcrt[i] = max(wcrt[i], T - releases[i])
                penalty[i] = max(penalty[i], T - releases[i] - deadlines[i])
                schedulable = False
            if t % task.period == 0:
                releases[i] = t
                durations[i] = task.duration
                deadlines[i] = t + task.deadline

        if any(v > 0 for v in durations):  # if there is some task to schedule:
            tmp = [dl if dr > 0 else 2 * T for dl, dr in zip(deadlines, durations)]
            ed_idx = tmp.index(min(tmp))
            timetable[t] = ed_idx
            durations[ed_idx] -= 1
            if durations[ed_idx] == 0 and deadlines[ed_idx] >= t:
                if t-releases[ed_idx] >= wcrt[ed_idx]:
                    wcrt[ed_idx] = t-releases[ed_idx]+1
        t += 1

    if any(v > 0 for v in durations):
        for idx in [i for i, response in enumerate(wcrt) if response == -1]:
            wcrt[idx] = max(wcrt[idx], T - releases[idx])
            penalty[idx] = max(penalty[idx], T - releases[idx])
            schedulable = False

    return schedulable, timetable, wcrt, sum(penalty) / (T * len(tasks))


def EDP(ps: PollingServer):
    delta = ps.period + ps.deadline - 2 * ps.duration
    alpha = ps.duration / ps.period

    T = lcm(*[obj.period for obj in ps.tasks])
    WCRT = [T] * len(ps.tasks)
    schedulable = True

    penalty = [0]*len(ps.tasks)

    for i, ETtask1 in enumerate(ps.tasks):
        t = 0
        while t <= T:
            supply = max(0, alpha * (t - delta))
            demand = 0
            for ETtask2 in ps.tasks:
                if ETtask2.priority >= ETtask1.priority:
                    demand += ceil(t / ETtask2.period) * ETtask2.duration

            if supply >= demand and t > 0:
                WCRT[i] = t
                break
            t += 1

        if WCRT[i] > ETtask1.deadline:
            penalty[i] = WCRT[i] - ETtask1.deadline
            schedulable = False
    return schedulable, WCRT, sum(penalty) / (T * len(ps.tasks))


def extention1(tasks):
    T = lcm(*[obj.period for obj in tasks])
    ## Di = relative deadline
    deadlines = [obj.deadline for obj in tasks]  # deadlines
    ## Ci = worst case excetion time
    __, __, wcrt, __ = EDF(tasks)
    ## Ti = period
    periods = [obj.period for obj in tasks]  # periods
    sum = 0
    for t in range(0, T):
        for i in range(0, len(deadlines)):
            sum += floor(((t + periods[i] - deadlines[i]) / periods[i])) * wcrt[i]
        if t > sum:
            return False
    return True


if __name__ == "__main__":
    path = "./test_cases/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"  # "./test_cases/taskset_small.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()
    params = [(5, 16, 5), (1, 4, 1), (50, 400, 131)]  # [(1070, 2000, 1580), (200, 2000, 1470), (100, 1000, 1000)]
    init_ps = []
    print("NUM ET", len(ET))
    max_sep = 2
    for idx, sep in enumerate(range(max_sep + 1)):
        tasks = [task for task in ET if task.separation == sep]
        budget, period, deadline = params[idx]
        init_ps.append(PollingServer("Polling Server", budget, period, deadline, tasks, separation=sep))

    print("EDP", [EDP(ps) for ps in init_ps])
    schedulable, timetable, wcrt, penalty = EDF(TT + init_ps)
    print("Extension1", extention1(TT + init_ps))
    print("EDF", schedulable, wcrt, penalty)
