import cpuinfo
from tqdm import tqdm

import config
import libraries.dataloader as dataloader
from libraries.algorithms import EDF
from libraries.graphplot import getTimetablePlot
from libraries.optimizers import GeneticAlgorithm, SimulatedAnnealing


def experiment(data_path, profiling=False):
    cpu = cpuinfo.get_cpu_info()
    print("{}, {} cores".format(cpu["brand_raw"], cpu["count"]))

    # load tasks
    dl = dataloader.DataLoader(data_path)
    TT, ET = dl.loadFile()

    if config.algorithm == "SA":
        optim = SimulatedAnnealing(
            TT,
            ET,
            numinstances=config.SA["numinstances"],
            numworkers=config.SA["numworkers"],
            maxiter=config.SA["maxiter"],
            toll=config.SA["toll"],
            convergence=config.SA["convergence"],
            extra_ps=config.SA["extra_ps"],
            wandblogging=config.SA["wandblogging"],
            iterationPerTemp=config.SA["iterationPerTemp"],
            initialTemp=config.SA["initialTemp"],
            finalTemp=config.SA["finalTemp"],
            tempReduction=config.SA["tempReduction"],
            alpha=config.SA["alpha"],
            beta=config.SA["beta"],
            dur_radius=config.SA["dur_radius"],
            dln_radius=config.SA["dln_radius"],
            priority_prob=config.SA["priority_prob"],
            free_tasks_switches=config.SA["free_tasks_switches"],
            no_upper_lim=config.SA["no_upper_lim"],
        )

    elif config.algorithm == "GA":
        optim = GeneticAlgorithm(
            TT,
            ET,
            numinstances=config.GA["numinstances"],
            numworkers=config.GA["numworkers"],
            maxiter=config.GA["maxiter"],
            toll=config.GA["toll"],
            convergence=config.SA["convergence"],
            wandblogging=config.GA["wandblogging"],
            pop_size=config.GA["pop_size"],
            num_parents=config.GA["num_parents"],
            p_cross=config.GA["p_cross"],
            p_mut=config.GA["p_mut"],
            selection=config.GA["selection"],
            free_tasks_switches=config.GA["free_tasks_switches"]
        )
    else:
        print(f"{config.algorithm} not recognized")
        quit()

    if config.SA["temptune"]:
        optim.plotTemperature().show()
        quit()

    bar_iter = optim.maxIter * optim.numInstances if optim.numWorkers == 1 else optim.numInstances

    if profiling:
        import cProfile
        import pstats

        with cProfile.Profile() as pr:
            with tqdm(
                total=bar_iter,
                desc="Progress",
                bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}",
            ) as bar:  # progress bar
                optim.run(bar.update)
        pr = pstats.Stats(pr)
        pr.sort_stats("cumulative").print_stats(10)
    else:
        with tqdm(
            total=bar_iter,
            desc="Progress",
            bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}",
        ) as bar:
            optim.run(bar.update)

    optim.printSolution()

    if config.show_plot:
        optim.plotCost(instance_idx="all").show()

        __, timetable, __, __ = EDF(TT + optim.bestSolution)
        getTimetablePlot(TT + optim.bestSolution, timetable, group_tt=True).show()

    if config.write_log:
        try:
            import os

            os.mkdir(config.log_directory + config.log_name)
        except FileExistsError:
            print(f"Overwriting {config.log_name} folder")

        cost_plt = optim.plotCost(instance_idx="all")
        cost_plt.savefig(config.log_directory + config.log_name + "/cost_plot.png")
        cost_plt.savefig(config.log_directory + config.log_name + "/cost_plot.eps")
        __, timetable, __, __ = EDF(TT + optim.bestSolution)
        timetable_plot = getTimetablePlot(TT + optim.bestSolution, timetable, group_tt=True)
        timetable_plot.savefig(config.log_directory + config.log_name + "/timetable_plot.png")
        timetable_plot.savefig(config.log_directory + config.log_name + "/timetable_plot.eps")

        with open(config.log_directory + config.log_name + "/log.txt", "w") as logfile:
            logfile.write("Platform: {}, {} cores\n\n".format(cpu["brand_raw"], cpu["count"]))
            logfile.write(f"Taskset: {config.test_case_path}\n\n")
            logfile.write(f"Algorithm: {optim.__class__.__name__}\n")
            logfile.write("Parameters:\n")

            for key, val in {
                "SimulatedAnnealing": config.SA,
                "GeneticAlgorithm": config.GA,
            }[optim.__class__.__name__].items():
                logfile.write(f"    {key}: {val}\n")
            logfile.write("\n" + optim.printSolution(get=True))


if __name__ == "__main__":
    experiment(config.test_case_path, config.profiling)
    print("All done")
