from libraries.algorithms import EDF, EDP
import libraries.dataloader as dataloader
from libraries.tasks import PollingServer
import random
import numpy as np
from sympy import divisors
from math import lcm
import itertools

class Optimizer:
    def __init__(self, TTtasks, ETtasks, maxiter=1000, toll=0.01):
        # Tasks
        self.TTtasks = TTtasks
        self.ETtasks = ETtasks

        # Termination Criteria
        self.maxIter = maxiter
        self.toll = toll

        # Periods
        self.hyperperiod = lcm(*[obj.period for obj in self.TTtasks])
        self.period_divisors = divisors(self.hyperperiod)[1:]

        # Utilities
        self.clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

        #Initialize solution
        self.solution = self.initializeSolution()
        self.currCost = self.computeCost(self.solution)
        self.currIter = 0

    def initializeSolution(self):
        max_sep = max([task.separation for task in self.ETtasks])
        solution = []
        for sep in range(1, max_sep+1):
            tasks = [task for task in self.ETtasks if task.separation==sep]
            period = random.choice(self.period_divisors)
            budget = random.randint(1, period)
            deadline = random.randint(budget, period)
            solution.append(PollingServer("PS {}".format(sep), budget, period, deadline, tasks, sep))
        return solution

    def isSatisfyingConstraints(self, solution) -> bool:
        cond = True
        for ps in solution:
            cond = cond and ps.duration <= ps.period and ps.duration <= ps.deadline <= ps.period
        return cond

    def isValidNeighbor(self, solution) -> bool:
        cond = self.isSatisfyingConstraints(solution)
        for ps in solution:
            cond = cond and EDP(ps)[0]
        cond = cond and EDF(self.TTtasks + solution)[0]
        return cond

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

    def printSolution(self):
        wcrt_tt = EDF(self.TTtasks + self.solution)[2]
        wcrt_et = []
        for ps in self.solution:
            print(ps)
            wcrt_et = wcrt_et + EDP(ps)[1]
        print("Solution cost: {:.5f}, valid: {}".format(self.computeCost(self.solution), self.isValidNeighbor(self.solution)))
        print("Average WCRT for TT+PS task: {:.2f} ms (max: {} ms)".format(np.mean(wcrt_tt), max(wcrt_tt)))
        print("Average WCRT for ET task: {:.2f} ms (max: {} ms)".format(np.mean(wcrt_et), max(wcrt_et)))

    def isTerminationCriteriaMet(self) -> bool:
        # Termination criteria
        return self.currIter >= self.maxIter or self.currCost < self.toll

    def run(self):
        while not self.isTerminationCriteriaMet():

            #update iteration counter
            self.currIter += 1

if __name__ == "__main__":
    path = "./test_cases/taskset_small.csv"
    #path = "./test_cases/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()

    optim = Optimizer(TT, ET, 10)
    optim.run()
    optim.printSolution()