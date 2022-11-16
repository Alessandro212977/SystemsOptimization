"""
Simulated Annealing Class
"""
import random
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from sympy import divisors
from math import lcm
import multiprocess as mp

from libraries.algorithms import EDF, EDP
import libraries.dataloader as dataloader
from libraries.tasks import PollingServer
from libraries.graphplot import plotTTtask

from alive_progress import alive_bar
import enlighten

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)  # DEBUG


class SimulatedAnnealing:
    def __init__(self, TTtasks, ETtasks, maxiter=1000, toll=0.01, iterationPerTemp=100, initialTemp=0.1, finalTemp=0.0001, tempReduction="geometric", alpha=0.5, beta=5):
        self.TTtasks = TTtasks
        self.ETtasks = ETtasks

        self.hyperperiod = lcm(*[obj.period for obj in self.TTtasks])
        self.period_divisors = divisors(self.hyperperiod)[1:]

        self.solution = self.initializeSolution()

        self.currCost = self.computeCost(self.solution)

        self.datalog = {
            "bins": [],
            "costs": [],
            "accepted_bins": []
        }

        self.clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

        self.currIter = 0
        self.maxIter = maxiter
        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        self.currIterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.beta = beta

        self.toll = toll

        self.decrementRule = {"linear": self.linearTempReduction, # t = t - a
                              "geometric": self.geometricTempReduction,  # t = t * a
                              "slowDecrease": self.slowDecreaseTempReduction}[tempReduction] # t = t / 1 + Bt

    def linearTempReduction(self):
        self.currTemp -= self.alpha

    def geometricTempReduction(self):
        self.currTemp *= self.alpha

    def slowDecreaseTempReduction(self):
        self.currTemp = self.currTemp / (1 + self.beta * self.currTemp)

    def isTerminationCriteriaMet(self):
        # can add more termination criteria
        return self.currTemp <= self.finalTemp or self.currIter >= self.maxIter or self.currCost < self.toll

    def initializeSolution(self):
        max_sep = max([task.separation for task in self.ETtasks])
        init_ps = []#[(300, 1000, 1000)]*max_sep   #[(1070, 2000, 1580), (200, 2000, 1470), (100, 1000, 1000)]
        for sep in range(1, max_sep+1):
            tasks = [task for task in self.ETtasks if task.separation==sep]
            period = random.choice(self.period_divisors)
            budget = random.randint(1, period)
            deadline = random.randint(budget, period)
            init_ps.append(PollingServer("PS {}".format(sep), budget, period, deadline, tasks, sep))
        return init_ps

    def run(self, pbar=None):
        while not self.isTerminationCriteriaMet():
            #print("Temperature: {}, prob: {}".format(self.currTemp,  math.exp(-0.1 / self.currTemp)))
            for i in range(self.currIterationPerTemp):
                # pick a random neighbor
                newSolution = self.neighborOperator(self.solution)

                # get the cost between the two solutions
                newcost = self.computeCost(newSolution)

                self.datalog["bins"].append(self.currIter)
                self.datalog["costs"].append(newcost)

                # if the new solution is better, accept it
                if newcost - self.currCost < 0 or self.acceptance(newcost - self.currCost):
                    #print("accepted sol. {}".format(self.isValidNeighbor(newSolution)))
                    self.solution = newSolution
                    self.currCost = newcost
                    self.datalog["accepted_bins"].append(self.currIter)

                # update progress bar
                if pbar:
                    pbar()

                #update iteration counter
                self.currIter += 1

            self.decrementRule()

    def plotCost(self):
        fig, ax = plt.subplots()
        #ax.plot(self.datalog["bins"], self.datalog["cost"])
        ax.plot(self.datalog["accepted_bins"], [self.datalog["costs"][i] for i in self.datalog["accepted_bins"]])#, color="red", marker="x")
        ax.grid()
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        ax.set_title("Cost of accepted solutions")
        plt.show()

    def printSolution(self):
        wcrt_tt = EDF(self.TTtasks + self.solution)[2]
        wcrt_et = []
        print("----------- Simulated Anealing Solution -----------")
        for ps in self.solution:
            print(ps)
            wcrt_et = wcrt_et + EDP(ps)[1]
        print("Solution cost: {:.5f}, valid: {}".format(self.computeCost(self.solution), self.isValidNeighbor(self.solution)))
        print("Average WCRT for TT+PS task: {:.2f} ms (max: {} ms)".format(np.mean(wcrt_tt), max(wcrt_tt)))
        print("Average WCRT for ET task: {:.2f} ms (max: {} ms)".format(np.mean(wcrt_et), max(wcrt_et)))
        print("---------------------------------------------------")

    def acceptance(self, cost):
        if cost > 0:
            # print("the solution could be accepted with prob: {:.3f}".format(math.exp(-cost/self.currTemp)))
            return random.random() < math.exp(-cost / self.currTemp)
        return False

    def computeCost(self, solution, params=["wcrt_tt", "wcrt_et"], weights=[0.5, 0.5]):
        assert sum(weights) == 1.0

        wcrt = []
        for ps in solution:
            wcrt = wcrt + EDP(ps)[1]

        params_dict = {
            "wcrt_tt": np.mean(EDF(self.TTtasks + solution)[2]),
            "wcrt_et": np.mean(wcrt),
            "duration": max([ps.duration for ps in solution]),
        }
        cost = 0 if self.isValidNeighbor(solution) else 1

        for w, val in zip(weights, params):
            cost += w * params_dict[val] / self.hyperperiod
        return cost

    def neighborOperator(self, center):
        def getrandomneighbor():
            new_center = []
            for ps in center:
                new_duration = self.clamp(ps.duration + random.randint(-10, 10) * 10, 1, self.hyperperiod)
                new_period = self.period_divisors[self.clamp(self.period_divisors.index(ps.period) + random.choice([-1, 0, 1]), 0, len(self.period_divisors)-1)]
                new_deadline = self.clamp(ps.deadline + random.randint(-10, 10) * 10, 1, self.hyperperiod)#self.period_divisors[self.period_divisors.index(ps.deadline) + random.choice([-1, 0, 1])]
                new_center.append(PollingServer(ps.name, new_duration, new_period, new_deadline, ps.tasks, ps.separation))
            return new_center

        neighbor = getrandomneighbor()
        while not self.isSatisfyingConstraints(neighbor):
            neighbor = getrandomneighbor()

        return neighbor

    def isSatisfyingConstraints(self, neighbor) -> bool:
        cond = True
        for ps in neighbor:
            cond = cond and ps.duration <= ps.period and ps.duration <= ps.deadline <= ps.period
        return cond

    def isValidNeighbor(self, neighbor) -> bool:
        cond = self.isSatisfyingConstraints(neighbor)
        for ps in neighbor:
            cond = cond and EDP(ps)[0]
        cond = cond and EDF(self.TTtasks + neighbor)[0]
        return cond

class MultiSimulatedAnealing:
    def __init__(self, sa_args, numworkers=None) -> None:
        self.numWorkers = numworkers
        self.sa_instances = [SimulatedAnnealing(*sa_args, maxiter=100) for __ in range(self.numWorkers)]
        self.maxIter = self.sa_instances[0].maxIter

    def run(self):
        print("----------- Parallel Anealing Solution ------------")
        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        print('{}, {} cores'.format(cpu['brand_raw'], cpu['count']))

        def func(args):
            with alive_bar(self.sa_instances[args].maxIter, theme="smooth", title="Process {}".format(args)) as bar:  # progress bar
                self.sa_instances[args].run(bar)

        with mp.Pool(mp.cpu_count()) as pool:
            for __ in pool.imap_unordered(func, [(i) for i in range(self.numWorkers)]):
                pass

        costs = [obj.currCost for obj in self.sa_instances]
        print("---------------------------------------------------")
        return self.sa_instances[np.argmin(costs)]

        

if __name__ == "__main__":
    path = "./test_cases/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"#"./test_cases/taskset_small.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()

    """
    msa = MultiSimulatedAnealing((TT, ET), 4)
    msa.run().printSolution()
    quit()
    """

    sa = SimulatedAnnealing(TT, ET)
    sa.printSolution()

    if False:
        import cProfile, pstats
        with cProfile.Profile() as pr:
            with alive_bar(sa.maxIter, theme="smooth") as bar:  # progress bar
                sa.run(bar)
        pr = pstats.Stats(pr)
        pr.sort_stats('cumulative').print_stats(10)
    else:
        with alive_bar(sa.maxIter, theme="smooth", title="Iterations:") as bar:  # progress bar
            sa.run(bar)

    schedulable_TT, timetable, wcrt_TT = EDF(TT + sa.solution)
    print(wcrt_TT)
    for ps in sa.solution:
        schedulable_ET, wcrt_ET = EDP(ps)
        print(wcrt_ET)

    sa.printSolution()
    sa.plotCost()

    plotTTtask(TT + sa.solution, timetable, group_tt=True)
