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
import matplotlib.pyplot as plt

class SimulatedAnnealing:
    def __init__(self, TTtasks, pollingserver, initialTemp=0.001, finalTemp=0, tempReduction='linear', iterationPerTemp=10, alpha=0.0001, beta=5):
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
        costs_plot = []
        accepted_costs_plot = []
        durations_plot = []
        bins = []
        bins_scatter = []

        while not self.isTerminationCriteriaMet():
            print("TEMPERATURE:", self.currTemp)
            for i in range(self.iterationPerTemp):
                # pick a random neighbor
                newSolution = self.neighborOperator(self.solution)
                schedule, responsetime = EDP(newSolution)
                sigma, wrct= EDF(self.TTtasks+[newSolution])
                while(schedule != True or not sigma):
                    newSolution = self.neighborOperator(self.solution)
                    schedule, responsetime = EDP(newSolution)
                    sigma, wrct= EDF(self.TTtasks+[newSolution])

                # get the cost between the two solutions
                sigma, wrctSolution= EDF(self.TTtasks+[self.solution])
                schedule, responsetime = EDP(self.solution)
                sigma, wrctNewSolution= EDF(self.TTtasks+[newSolution])
                schedule, responsetimeNew = EDP(newSolution)

                cost = self.ComputeCost(wrctNewSolution, responsetimeNew) - self.ComputeCost(wrctSolution, responsetime)
                print("cost: {}, new duration: {}, new period {}, acceptance prob: {}".format(cost, newSolution.duration, newSolution.period, math.exp(cost / self.currTemp)))
                costs_plot.append(self.ComputeCost(wrctNewSolution, responsetimeNew))
                durations_plot.append(newSolution.duration)
                bins.append(bins[-1]+1 if bins else 0)
                
                # if the new solution is better, accept it
                if cost < 0 or self.acceptance(cost):
                    self.solution = newSolution
                    accepted_costs_plot.append(self.ComputeCost(wrctNewSolution, responsetimeNew))
                    bins_scatter.append(bins[-1])


            self.decrementRule()

        print(responsetime)
        plt.plot(bins, costs_plot)
        plt.scatter(bins_scatter, accepted_costs_plot)
        #plt.plot(bins[1:], durations_plot)
        plt.show()

    def acceptance(self, cost):
        if cost > 0:
            print("the solution is accepted with prob", math.exp(-cost/self.currTemp))
            return random.random() < math.exp(-cost/self.currTemp)
        return False

    def ComputeCost(self, wcrt, response, weight=0.6):
        #wcrt_norm = preprocessing.normalize([np.array([np.mean(wcrtNew), np.mean(wcrtOld)])])[0]
        #print(np.mean(wcrtNew), np.mean(wcrtOld), wcrt_norm)
        #response_norm = preprocessing.normalize([np.array([responseNew, responseOld])])[0]

        norm = preprocessing.normalize([np.array([np.mean(wcrt), response])])[0]
        return weight*np.mean(norm[0]) + (1-weight)*norm[1]
        return (response_norm[0] - response_norm[1])
        return (wcrt_norm[0] - wcrt_norm[1]) #+ (1-weight)*(response_norm[0] - response_norm[1])

    def neighborOperator(self, center):
        duration_offset = random.randint(-20, 20)#*10
        period_offset = 0#random.randint(-3, 3)*100
        return PollingServer(center.duration + duration_offset, center.period + period_offset, center.deadline, center.tasks)

if __name__ == "__main__":
    path = "./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = dataloader.DataLoader(path)
    TT, ET  = dl.loadFile()
    TT, ET = TT[:10], ET[:10]
    init_ps = PollingServer(1800, 2000, 2000, ET)
    sa = SimulatedAnnealing(TT, init_ps)
    sa.run()





