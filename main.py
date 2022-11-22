from tqdm import tqdm

import libraries.dataloader as dataloader
from libraries.optimizers import SimulatedAnnealing, GeneticAlgorithm
from libraries.algorithms import EDF, EDP
from libraries.graphplot import getTimetablePlot


def SA(TT, ET):
    sa = SimulatedAnnealing(TT, ET, maxiter=100)
    # sa.plotTemperature()
    sa.printSolution()

    if False:
        import cProfile, pstats

        with cProfile.Profile() as pr:
            with tqdm(total=sa.maxIter, desc="Iterations") as bar:  # progress bar
                sa.run(bar.update)
        pr = pstats.Stats(pr)
        pr.sort_stats("cumulative").print_stats(10)
    else:
        with tqdm(total=sa.maxIter, desc="Iterations") as bar:
            sa.run(bar.update)

    schedulable_TT, timetable, wcrt_TT, __ = EDF(TT + sa.solution)
    print(wcrt_TT)
    for ps in sa.solution:
        schedulable_ET, wcrt_ET, __ = EDP(ps)
        print(wcrt_ET)

    sa.printSolution()
    sa.plotCost()

    getTimetablePlot(TT + sa.solution, timetable, group_tt=True).show()


def GA(TT, ET):
    ga = GeneticAlgorithm(TT, ET, 10)
    ga.printSolution()

    if False:
        import cProfile, pstats

        with cProfile.Profile() as pr:
            with tqdm(total=ga.maxIter, desc="Iterations") as bar:  # progress bar
                ga.run(bar.update)
        pr = pstats.Stats(pr)
        pr.sort_stats("cumulative").print_stats(10)
    else:
        with tqdm(total=ga.maxIter, desc="Iterations") as bar:
            ga.run(bar.update)

    schedulable_TT, timetable, wcrt_TT, __ = EDF(TT + ga.solution)
    print(wcrt_TT)
    for ps in ga.solution:
        schedulable_ET, wcrt_ET, __ = EDP(ps)
        print(wcrt_ET)

    ga.printSolution()
    ga.plotCost()

    getTimetablePlot(TT + ga.solution, timetable, group_tt=True).show()


def main():
    import cpuinfo
    cpu = cpuinfo.get_cpu_info()
    print("{}, {} cores".format(cpu["brand_raw"], cpu["count"]))
    
    path = "./test_cases/taskset_small.csv"
    # path = "./test_cases/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()

    # SA(TT, ET)
    GA(TT, ET)


if __name__ == "__main__":
    main()
