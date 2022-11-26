from tqdm import tqdm

import libraries.dataloader as dataloader
from libraries.optimizers import (
    SimulatedAnnealing,
    GeneticAlgorithm,
)
from libraries.algorithms import EDF, EDP
from libraries.graphplot import getTimetablePlot

def experiment(data_path, profiling=False):

    # load tasks
    dl = dataloader.DataLoader(data_path)
    TT, ET = dl.loadFile()

    #optim = SimulatedAnnealing(TT, ET, numinstances=4, numworkers=4, maxiter=200)
    optim = GeneticAlgorithm(TT, ET, numinstances=2, numworkers=2, maxiter=50, pop_size=16, num_parents=4)
    # optim.plotTemperature()

    if profiling:
        import cProfile, pstats

        with cProfile.Profile() as pr:
            with tqdm(total=optim.maxIter, desc="Iterations") as bar:  # progress bar
                optim.run(bar.update)
        pr = pstats.Stats(pr)
        pr.sort_stats("cumulative").print_stats(10)
    else:
        with tqdm(total=optim.maxIter, desc="Iterations") as bar:
            optim.run(bar.update)


    optim.printSolution()
    optim.plotBars()
    optim.plotCost(instance_idx="all")

    __, timetable, __, __ = EDF(TT + optim.bestSolution)
    getTimetablePlot(TT + optim.bestSolution, timetable, group_tt=True).show()


if __name__ == "__main__":
    import cpuinfo
    cpu = cpuinfo.get_cpu_info()
    print("{}, {} cores".format(cpu["brand_raw"], cpu["count"]))

    path = "./test_cases/taskset_small.csv"
    # path = "./test_cases/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    experiment(path)
