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

from libraries.algorithms import EDF, EDP
import libraries.dataloader as dataloader
from libraries.tasks import PollingServer
from libraries.graphplot import plotTTtask

from alive_progress import alive_bar

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)  # DEBUG


class SimulatedAnnealing:
    def __init__(self, TTtasks, pollingservers, initialTemp=0.001, finalTemp=0.00001, tempReduction="linear", iterationPerTemp=10, alpha=0.0001, beta=5):
        self.TTtasks = TTtasks

        self.solution = pollingservers
        if not self.isValidNeighbor(self.solution):
            print("Warning: starting from invalid solution")

        self.hyperperiod = lcm(*[obj.period for obj in self.TTtasks])
        self.period_divisors = divisors(self.hyperperiod)

        self.currCost = self.computeCost(self.solution)

        self.datalog = {
            "bins": [],
            "costs": [],
            "accepted_bins": [],
            "duration": [],
            "period": [],
            "deadline": [],
        }

        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        self.iterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.beta = beta

        if tempReduction == "linear":
            self.decrementRule = self.linearTempReduction
            # t = t - a
        elif tempReduction == "geometric":
            self.decrementRule = self.geometricTempReduction
            # t = t * a
        elif tempReduction == "slowDecrease":
            self.decrementRule = self.slowDecreaseTempReduction
            # t = t / 1 + Bt
        else:
            self.decrementRule = tempReduction

    def linearTempReduction(self):
        self.currTemp -= self.alpha

    def geometricTempReduction(self):
        self.currTemp *= self.alpha

    def slowDecreaseTempReduction(self):
        self.currTemp = self.currTemp / (1 + self.beta * self.currTemp)

    def isTerminationCriteriaMet(self):
        # can add more termination criteria
        return self.currTemp <= self.finalTemp

    def run(self, pbar=None):
        while not self.isTerminationCriteriaMet():
            logging.debug("Temperature: {}".format(self.currTemp))
            for i in range(self.iterationPerTemp):
                # pick a random neighbor
                newSolution = self.neighborOperator(self.solution)

                # get the cost between the two solutions
                newcost = self.computeCost(newSolution)

                logging.debug("Iter: {}, cost: {:.3f}, new duration: {}, new period: {}, new deadline: {}".format(len(self.datalog["bins"]), newcost, newSolution[0].duration, newSolution[0].period, newSolution[0].deadline))
                self.datalog["bins"].append(self.datalog["bins"][-1] + 1 if self.datalog["bins"] else 0)
                self.datalog["costs"].append(newcost)
                self.datalog["duration"].append(newSolution[0].duration)
                self.datalog["period"].append(newSolution[0].period)
                self.datalog["deadline"].append(newSolution[0].deadline)

                # if the new solution is better, accept it
                if newcost - self.currCost < 0 or self.acceptance(newcost - self.currCost):
                    self.solution = newSolution
                    self.currCost = newcost
                    self.datalog["accepted_bins"].append(self.datalog["bins"][-1])

                # update progress bar
                if pbar:
                    pbar()

            self.decrementRule()

    def plotSolution(self, feature_list=["costs", "duration", "period", "deadline"]):
        for feature in feature_list:
            fig, ax = plt.subplots()
            ax.plot(self.datalog["bins"], self.datalog[feature])
            ax.scatter(self.datalog["accepted_bins"], [self.datalog[feature][i] for i in self.datalog["accepted_bins"]], color="red", marker="x")
            ax.grid()
            ax.set_xlabel("Iterations")
            ax.set_ylabel(feature)
            ax.set_title(feature)
        plt.show()

    def printSolution(self):
        print(self.solution[0])

    def acceptance(self, cost):
        if cost > 0:
            # print("the solution could be accepted with prob: {:.3f}".format(math.exp(-cost/self.currTemp)))
            return random.random() < math.exp(-cost / self.currTemp)
        return False

    def computeCost(self, solution, params=["wcrt_tt", "wcrt_et", "duration"], weights=[0.4, 0.4, 0.2]):
        assert sum(weights) == 1.0

        wcrt = []
        for ps in solution:
            wcrt = wcrt + EDP(ps)[1]

        params_dict = {
            "wcrt_tt": np.mean(EDF(self.TTtasks + solution)[2]),
            "wcrt_et": np.mean(wcrt),
            "duration": max([ps.duration for ps in solution]),
        }
        cost = 0

        for w, val in zip(weights, params):
            cost += w * params_dict[val] / self.hyperperiod
        return cost

    def neighborOperator(self, center):
        def getrandomneighbor():
            new_center = []
            for ps in center:
                new_duration = ps.duration + random.randint(-10, 10) * 10
                new_period = self.period_divisors[self.period_divisors.index(ps.period) + random.choice([-1, 0, 1])]
                new_deadline = self.period_divisors[self.period_divisors.index(ps.deadline) + random.choice([-1, 0, 1])]
                new_center.append(PollingServer(ps.name, new_duration, new_period, new_deadline, ps.tasks))
            return new_center

        neighbor = getrandomneighbor()
        while not self.isValidNeighbor(neighbor):
            neighbor = getrandomneighbor()

        return neighbor

    def isValidNeighbor(self, neighbor) -> bool:
        cond = True
        for ps in neighbor:
            cond = cond and ps.duration <= ps.period and ps.deadline <= ps.period and EDP(ps)[0]
        cond = cond and EDF(self.TTtasks + neighbor)[0]
        return cond


if __name__ == "__main__":
    path = "./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()
    #TT, ET = TT, ET[:5]
    init_ps = [PollingServer("Polling Server", 1200, 1500, 1500, ET)]

    sa = SimulatedAnnealing(TT, init_ps)
    #import cProfile
    #with cProfile.Profile() as pr:
    with alive_bar(sa.iterationPerTemp * 10, theme="smooth") as bar:  # progress bar
        sa.run(bar)
    #pr.print_stats()

    schedulable_TT, timetable, wcrt_TT = EDF(TT + sa.solution)
    schedulable_ET, wcrt_ET = EDP(sa.solution[0])

    print(wcrt_TT, wcrt_ET)

    sa.printSolution()
    sa.plotSolution()

    plotTTtask(TT + sa.solution, timetable)
