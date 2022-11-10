"""
Simulated Annealing Class
"""
import random
import math
import numpy as np
from libraries.algorithms import EDF, EDP
import libraries.dataloader as dataloader
from libraries.tasks import PollingServer
import matplotlib.pyplot as plt
from sympy import divisors
from math import lcm
from libraries.graphplot import plotTTtask

class SimulatedAnnealing:
    def __init__(self, TTtasks, pollingserver, initialTemp=0.001, finalTemp=0.00001, tempReduction='linear', iterationPerTemp=10, alpha=0.0001, beta=5):
        self.solution = pollingserver

        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        self.iterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.beta = beta

        self.TTtasks = TTtasks

        self.hyperperiod = lcm(*[obj.period for obj in self.TTtasks])
        self.period_divisors = divisors(self.hyperperiod)

        self.datalog = {"bins": [],
                        "costs": [],
                        "accepted_bins": [],
                        "duration": [],
                        "period": [],
                        "deadline": [],
                        "responsetime": []}

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

    def run(self):
        while not self.isTerminationCriteriaMet():
            print("TEMPERATURE:", self.currTemp)
            for i in range(self.iterationPerTemp):
                # pick a random neighbor
                newSolution = self.neighborOperator(self.solution)
                
                # get the cost between the two solutions
                __, wrctSolution= EDF(self.TTtasks+[self.solution])
                __, responsetime = EDP(self.solution)
                __, wrctNewSolution= EDF(self.TTtasks+[newSolution])
                __, responsetimeNew = EDP(newSolution)

                cost = self.computeCost(wrctNewSolution, responsetimeNew, newSolution.duration) - self.computeCost(wrctSolution, responsetime, self.solution.duration)
                print("Iter: {}, cost: {:.3f}, new duration: {}, new period: {}, new deadline: {}".format(len(self.datalog["bins"]), self.computeCost(wrctNewSolution, responsetimeNew, newSolution.duration), newSolution.duration, newSolution.period, newSolution.deadline))
                self.datalog["bins"].append(self.datalog["bins"][-1] + 1 if self.datalog["bins"] else 0)
                self.datalog["costs"].append(self.computeCost(wrctNewSolution, responsetimeNew, newSolution.duration))
                self.datalog["responsetime"].append(responsetimeNew)
                self.datalog["duration"].append(newSolution.duration)
                self.datalog["period"].append(newSolution.period)
                self.datalog["deadline"].append(newSolution.deadline)

                # if the new solution is better, accept it
                if cost < 0 or self.acceptance(cost):
                    self.solution = newSolution
                    self.datalog["accepted_bins"].append(self.datalog["bins"][-1])

            self.decrementRule()
    
    def plotSolution(self, feature_list=["costs", "responsetime", "duration", "period", "deadline"]):
        for feature in feature_list:
            fig, ax = plt.subplots()
            ax.plot(self.datalog["bins"], self.datalog[feature])
            ax.scatter(self.datalog["accepted_bins"], [self.datalog[feature][i] for i in self.datalog["accepted_bins"]], color='red', marker="x")
            ax.grid()
            ax.set_xlabel("Iterations")
            ax.set_ylabel(feature)
            ax.set_title(feature)
        plt.show()

    def printSolution(self):
        print(self.solution)

    def acceptance(self, cost):
        if cost > 0:
            #print("the solution could be accepted with prob: {:.3f}".format(math.exp(-cost/self.currTemp)))
            return random.random() < math.exp(-cost/self.currTemp)
        return False

    def computeCost(self, wcrt, response, duration, weight=0.5):
        return weight*np.mean(wcrt)/self.hyperperiod + (1-weight)*response/self.hyperperiod# + 0.3*np.mean(duration/self.hyperperiod)

    def neighborOperator(self, center):
        def getrandomneighbor():
            new_duration = center.duration + random.randint(-10, 10)*10
            new_period = self.period_divisors[self.period_divisors.index(center.period) + random.choice([-1, 0, 1])]
            new_deadline = self.period_divisors[self.period_divisors.index(center.deadline) + random.choice([-1, 0, 1])]
            return PollingServer(center.name, new_duration, new_period, new_deadline, center.tasks)

        neighbor = getrandomneighbor()
        while(not self.isValidNeighbor(neighbor)):
            neighbor = getrandomneighbor()
       
        return neighbor
    
    def isValidNeighbor(self, neighbor) -> bool:
        schedule, responsetime = EDP(neighbor)
        sigma, wrct= EDF(self.TTtasks+[neighbor])
        return schedule and sigma and neighbor.duration <= neighbor.period

if __name__ == "__main__":
    path = "./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = dataloader.DataLoader(path)
    TT, ET  = dl.loadFile()
    TT, ET = TT, ET[:5]
    init_ps = PollingServer("Polling Server", 1300, 2000, 2000, ET)

    sa = SimulatedAnnealing(TT, init_ps)
    sa.run()

    sigma,  wrctNewSolution= EDF(TT+[sa.solution])
    schedule, responsetimeNew = EDP(sa.solution)
    print(wrctNewSolution, schedule, responsetimeNew)

    sa.printSolution()
    sa.plotSolution()

    plotTTtask(TT+[sa.solution], sigma)






