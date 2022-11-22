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
from tqdm import tqdm 

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

    def plotTemperature(self):
        iterations = list(range(int(self.maxIter/self.currIterationPerTemp)+1))
        temperatures = [self.currTemp*self.alpha**i for i in iterations]
        differences = [0.001, 0.01, 0.1, 1.0]
        for d in differences:
            metropolis = [np.exp(-d/t) for t in temperatures]
            # plot iterations vs metropolis
            label = 'diff=%.3f' % d
            plt.plot(iterations, metropolis, marker='x', label=label)
        # inalize plot
        plt.xlabel('Iteration')
        plt.ylabel('Metropolis Criterion')
        #plt.ylim((0.0001, 1))
        #plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.show()

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
                if -cost > 0:
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
        ax.set_ylim((0, 3))
        ax.grid()
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        ax.set_title("Cost of accepted solutions")
        plt.show()

    def printSolution(self):
        wcrt_tt = EDF(self.TTtasks + self.solution)[2]
        wcrt_et = []
        print("----------- Simulated Annealing Solution -----------")
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

    def computeCost(self, solution, weights=[0.5, 0.5]):
        assert sum(weights) == 1.0

        #EDP
        et_schedulable, et_wcrt, et_penalty = True, [], 0
        for ps in solution:
            schedulable, WCRT, penalty = EDP(ps)
            et_schedulable = et_schedulable and schedulable
            et_wcrt = et_wcrt + WCRT
            et_penalty += penalty
        et_penalty /= len(solution)

        #EDF
        tt_schedulable, __, tt_wcrt, tt_penalty = EDF(self.TTtasks + solution)

        cost = 0 if et_schedulable and tt_schedulable else 1 + 0.5 * et_penalty + 0.5 * tt_penalty
        for w, c in zip(weights, [np.mean(et_wcrt), np.mean(tt_wcrt)]):
            cost += w * c / self.hyperperiod
        return cost

    def neighborOperator(self, center):
        def getrandomneighbor():
            new_center = []
            for ps in center:
                new_period = self.period_divisors[self.clamp(self.period_divisors.index(ps.period) + random.choice([-1, 0, 1]), 0, len(self.period_divisors)-1)]
                new_duration = self.clamp(ps.duration + random.randint(-10, 10) * 10, 1, new_period)
                new_deadline = self.clamp(ps.deadline + random.randint(-10, 10) * 10, new_duration, new_period)
                new_center.append(PollingServer(ps.name, new_duration, new_period, new_deadline, ps.tasks, ps.separation))
            return new_center

        neighbor = getrandomneighbor()
        #while not self.isSatisfyingConstraints(neighbor):
        #    neighbor = getrandomneighbor()
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

class MultiSimulatedAnnealing:
    def __init__(self, sa_args, numworkers=None) -> None:
        self.numWorkers = numworkers
        self.sa_instances = [SimulatedAnnealing(*sa_args, maxiter=50) for __ in range(self.numWorkers)]
        self.maxIter = self.sa_instances[0].maxIter

    def run(self):
        print("----------- Parallel Annealing Solution ------------")
        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        print('{}, {} cores'.format(cpu['brand_raw'], cpu['count']))
        costs = [obj.currCost for obj in self.sa_instances]
        print("before", costs)
        print()

        def func(args):
            with tqdm(total=self.sa_instances[args].maxIter, position=args, desc="Process: {}".format(args), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=True) as bar:
                self.sa_instances[args].run(bar.update)
            return self.sa_instances[args]

        sa = []
        with mp.Pool(mp.cpu_count()) as pool:
            for sol in pool.imap_unordered(func, [(i) for i in range(self.numWorkers)]):
                sa.append(sol)

        costs = [obj.currCost for obj in sa]
        print()
        print("after", costs)
        print("---------------------------------------------------")
        self.solution = sa[np.argmin(costs)]
        return self.solution

        

if __name__ == "__main__":
    path = "./test_cases/taskset_small.csv"
    #path = "./test_cases/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()


    #"""
    sa = SimulatedAnnealing(TT, ET, maxiter=1000)
    #sa.plotTemperature()
    sa.printSolution()

    if False:
        import cProfile, pstats
        with cProfile.Profile() as pr:
            with tqdm(total=sa.maxIter, desc="Iterations") as bar:  # progress bar
                sa.run(bar.update)
        pr = pstats.Stats(pr)
        pr.sort_stats('cumulative').print_stats(10)
    else:
        with tqdm(total=sa.maxIter, desc="Iterations") as bar:
            sa.run(bar.update)
    """

    msa = MultiSimulatedAnnealing((TT, ET), 8)
    sa = msa.run()
    """

    schedulable_TT, timetable, wcrt_TT, __ = EDF(TT + sa.solution)
    print(wcrt_TT)
    for ps in sa.solution:
        schedulable_ET, wcrt_ET, __ = EDP(ps)
        print(wcrt_ET)

    sa.printSolution()
    sa.plotCost()

    plotTTtask(TT + sa.solution, timetable, group_tt=True)
