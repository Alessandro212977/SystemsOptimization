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
    def __init__(self, TTtasks, pollingservers, initialTemp=0.001, finalTemp=0.00001, tempReduction="linear", iterationPerTemp=100, maxiter=1000, alpha=0.000001, beta=5):
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

        self.iter = 0
        self.MaxIter = maxiter
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
        return self.currTemp <= self.finalTemp or self.iter > self.MaxIter

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
                    #print("accepted sol. {}".format(self.isValidNeighbor(newSolution)))
                    self.solution = newSolution
                    self.currCost = newcost
                    self.datalog["accepted_bins"].append(self.datalog["bins"][-1])

                # update progress bar
                if pbar:
                    pbar()

                self.iter += 1

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
        for ps in self.solution:
            print(ps)

    def acceptance(self, cost):
        if cost > 0:
            # print("the solution could be accepted with prob: {:.3f}".format(math.exp(-cost/self.currTemp)))
            return random.random() < math.exp(-cost / self.currTemp)
        return False

    def computeCost(self, solution, params=["wcrt_tt", "wcrt_et", "duration"], weights=[0.5, 0.5, 0]):
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
            #print("divisors", self.period_divisors)
            #print("ps period", center[0].period)
            for ps in center:
                new_duration = ps.duration + random.randint(-10, 10) * 10
                new_period = self.period_divisors[self.period_divisors.index(ps.period) + random.choice([-1, 0, 1])]
                new_deadline = ps.deadline + random.randint(-10, 10) * 10#self.period_divisors[self.period_divisors.index(ps.deadline) + random.choice([-1, 0, 1])]
                new_center.append(PollingServer(ps.name, new_duration, new_period, new_deadline, ps.tasks, ps.separation))
            return new_center

        neighbor = getrandomneighbor()
        while not self.isValidNeighbor(neighbor):
            neighbor = getrandomneighbor()

        return neighbor

    def isValidNeighbor(self, neighbor) -> bool:
        cond = True
        for ps in neighbor:
            #print("ps", ps, cond)
            cond = cond and ps.duration <= ps.period and ps.deadline <= ps.period# and EDP(ps)[0]
            #print("after edp ps", ps, cond)
        #cond = cond and EDF(self.TTtasks + neighbor)[0]
        #print("after edf schedulable", EDF(self.TTtasks + neighbor)[0])
        return cond


if __name__ == "__main__":
    path = "./test_cases/taskset_small.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()
    #TT, ET = TT, ET[:5]
    max_sep = max([task.separation for task in ET])
    init_ps = []
    params = [(300, 1000, 1000), (300, 1000, 1000), (300, 1000, 1000)]#[(1070, 2000, 1580), (200, 2000, 1470), (100, 1000, 1000)]
    for idx, sep in enumerate(range(1, max_sep+1)):
        tasks = [task for task in ET if task.separation==sep]
        budget, period, deadline = params[idx]
        init_ps.append(PollingServer("Polling Server {}".format(sep), budget, period, deadline, tasks, separation=sep))

    for ps in init_ps:
        print(ps)

    sa = SimulatedAnnealing(TT, init_ps)
    #import cProfile
    #with cProfile.Profile() as pr:
    with alive_bar(sa.iterationPerTemp * 10, theme="smooth") as bar:  # progress bar
        sa.run(bar)
    #pr.print_stats()


    schedulable_TT, timetable, wcrt_TT = EDF(TT + sa.solution)
    print(wcrt_TT)
    for ps in sa.solution:
        schedulable_ET, wcrt_ET = EDP(ps)
        print(wcrt_ET)

    sa.printSolution()
    sa.plotSolution()

    plotTTtask(TT + sa.solution, timetable)
