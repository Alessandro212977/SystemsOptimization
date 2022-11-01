"""
Simulated Annealing Class
"""
import random
import math
from pollingserver import PollingServer as ps
from ETalgorithm import EDP
from EDFalgorithm import EDF

class SimulatedAnnealing:
    def __init__(self, pollingserver, initialTemp, finalTemp, tempReduction, neighborOperator,TTtasks , iterationPerTemp=100, alpha=10, beta=5):
        self.solution = pollingserver
        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        self.iterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.beta = beta
        self.neighborOperator = neighborOperator
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
                neighbors = self.neighborOperator(self.solution, self.TTtasks)
                # pick a random neighbor
                newSolution = random.choice(neighbors)
                # get the cost between the two solutions
                sigma, wrctSolution= EDF([self.solution],self.TTtasks)
                sigma, wrctNewSolution= EDF([newSolution],self.TTtasks)
                cost = wrctSolution - wrctNewSolution
                # if the new solution is better, accept it
                if cost >= 0:
                    self.solution = newSolution
                # if the new solution is not better, accept it with a probability of e^(-cost/temp)
                else:
                    if random.uniform(0, 1) < math.exp(-cost / self.currTemp):
                        self.solution = newSolution
            # decrement the temperature
            self.decrementRule()

    def neighborOperator(self):
        neighbors = []
        for i in range (100, 10000, 100):
            neighbor =ps(self.solution.duration + i, self.solution.period + i, self.solution.deadline + i, self.solution.tasks)
            schedule,responsetime = EDP(neighbor)
            sigma, wrct= EDF([neighbor],self.TTtasks)
            if schedule == True and sigma:
                neighbors.append(neighbor)
        return neighbors





