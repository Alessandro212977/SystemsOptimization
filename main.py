# Hellow World
from libraries.algorithms import EDF, EDP
import libraries.dataloader as dataloader
from libraries.tasks import PollingServer
import numpy as np

def main():
    path = "./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = dataloader.DataLoader(path)
    TT, ET  = dl.loadFile()

    durations = list(range(100, 3000, 100))
    periods = list(range(1900, 2000, 100))

    for d in durations:
        for p in periods:
            print(d, p)
            ps = PollingServer(duration=d, period=p, deadline=2000, tasks=ET)
            schedulable, responseTime, sigma, WCRT = EDP(ps) + EDF(TT+[ps])
            if schedulable and sigma != []:
                print("ok", d, p, np.mean(WCRT), responseTime)


if __name__ == "__main__":
    main()