"""
Simulated Annealing Class
"""
import random
import math
import numpy as np
from libraries.algorithms import EDF, EDP
import libraries.dataloader as dataloader
from libraries.tasks import PollingServer
from sklearn import preprocessing

class SimulatedAnnealing:
    def __init__(self, TTtasks, pollingserver, initialTemp=0.2, finalTemp=0, tempReduction='linear', iterationPerTemp=10, alpha=0.01, beta=5):
        self.solution = pollingserver
        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        self.iterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.beta = beta
        self.TTtasks = TTtasks

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
        return self.currTemp <= self.finalTemp or self.neighborOperator(self.solution) == 0

    def run(self):
        while not self.isTerminationCriteriaMet():
            # iterate that number of times
            for i in range(self.iterationPerTemp):
                # get all of the neighbors
                neighbors = self.neighborOperator(self.solution)
                # pick a random neighbor
                newSolution = random.choice(neighbors)
                schedule, responsetime = EDP(newSolution)
                sigma, wrct= EDF(self.TTtasks+[newSolution])
                while(schedule != True or not sigma):
                    neighbors.remove(newSolution)
                    newSolution = random.choice(neighbors)
                    schedule, responsetime = EDP(newSolution)
                    sigma, wrct= EDF(self.TTtasks+[newSolution])

                # get the cost between the two solutions
                sigma, wrctSolution= EDF(self.TTtasks+[self.solution])
                schedule, responsetime = EDP(self.solution)
                sigma, wrctNewSolution= EDF(self.TTtasks+[newSolution])
                schedule, responsetimeNew = EDP(newSolution)
                wcrt_norm = preprocessing.normalize([np.array([np.mean(wrctSolution), np.mean(wrctNewSolution)])])[0]
                response_norm = preprocessing.normalize([np.array([responsetime, responsetimeNew])])[0]
                cost = 0.6*(wcrt_norm[0] - wcrt_norm[1]) +0.4*(response_norm[0] - response_norm[1])
                # if the new solution is better, accept it
                if cost > 0:
                    self.solution = newSolution
                    print("this solution is better than the old one")
                # if the new solution is not better, accept it with a probability of e^(-cost/temp)
                elif random.uniform(0, 1) < math.exp(-cost / self.currTemp):
                    print("this solution is not better but it got accepted prob:", math.exp(-cost / self.currTemp), "temp:", self.currTemp)
                    self.solution = newSolution

                print(i, self.solution, "cost:", cost)
            # decrement the temperature
            self.decrementRule()

    def neighborOperator(self, center):
        neighbors = []
        for i in [-10, 10]:
            #for j in [-10, 10]:
            #    for k in [-10, 10]:
            neighbor = PollingServer(center.duration + i, center.period, center.deadline, center.tasks)
            neighbors.append(neighbor)
        return neighbors

if __name__ == "__main__":
    path = "./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = dataloader.DataLoader(path)
    TT, ET  = dl.loadFile()
    TT, ET = TT[:3], ET[:3]
    init_ps = PollingServer(1800, 2000, 2000, ET)
    sa = SimulatedAnnealing(TT, init_ps)
    sa.run()





