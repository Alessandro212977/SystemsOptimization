import itertools
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

class Optimizer:
    def __init__(self, TTtasks, ETtasks, maxiter=100, toll=0.1):
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

class SimulatedAnnealing(Optimizer):
    def __init__(self, TTtasks, ETtasks, maxiter=1000, toll=0.01, iterationPerTemp=100, initialTemp=0.1, finalTemp=0.0001, tempReduction="geometric", alpha=0.5, beta=5):
        super().__init__(TTtasks, ETtasks, maxiter, toll)

        self.datalog = {
            "bins": [],
            "costs": [],
            "accepted_bins": []
        }

        self.currTemp = initialTemp
        self.finalTemp = finalTemp
        self.currIterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.beta = beta

        self.decrementRule = {"linear": self.linearTempReduction, # t = t - a
                              "geometric": self.geometricTempReduction,  # t = t * a
                              "slowDecrease": self.slowDecreaseTempReduction}[tempReduction] # t = t / 1 + Bt

    def linearTempReduction(self):
        self.currTemp -= self.alpha

    def geometricTempReduction(self):
        self.currTemp *= self.alpha

    def slowDecreaseTempReduction(self):
        self.currTemp = self.currTemp / (1 + self.beta * self.currTemp)

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

    def acceptance(self, cost):
        if cost > 0:
            # print("the solution could be accepted with prob: {:.3f}".format(math.exp(-cost/self.currTemp)))
            return random.random() < math.exp(-cost / self.currTemp)
        return False

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

    def run(self, pbar=None):
        while not self.isTerminationCriteriaMet():
            #print("Temperature: {}, prob: {}".format(self.currTemp,  math.exp(-0.1 / self.currTemp)))
            for i in range(self.currIterationPerTemp):
                # pick a random neighbor
                newSolution = self.neighborOperator(self.solution)

                # get the cost between the two solutions
                newCost = self.computeCost(newSolution)

                self.datalog["bins"].append(self.currIter)
                self.datalog["costs"].append(newCost)

                # if the new solution is better, accept it
                if newCost - self.currCost < 0 or self.acceptance(newCost - self.currCost):
                    #print("accepted sol. {}".format(self.isValidNeighbor(newSolution)))
                    self.solution = newSolution
                    self.currCost = newCost
                    self.datalog["accepted_bins"].append(self.currIter)

                # update progress bar
                if pbar:
                    pbar()

                #update iteration counter
                self.currIter += 1

            self.decrementRule()

class GeneticAlgorithm(Optimizer):
    def __init__(self, TTtasks, ETtasks, maxiter=100, toll=0.01, pop_size=10, p_cross=0.8, p_mut=0.2):
        super().__init__(TTtasks, ETtasks, maxiter, toll)

        self.popSize = pop_size
        self.pCross = p_cross
        self.pMut = p_mut

    def initialPopulation(self):
        pop = []
        for __ in range(self.popSize):
            pop.append(self.initializeSolution())
        return pop

    def evaluatePopulation(self, population, weights=[0.5, 0.5]):
        costs_list = []
        for solution in population:
            costs_list.append(self.computeCost(solution, weights))
        return costs_list

    def paramsToList(self, solution):
        params = []
        for ps in solution:
            params.append(ps.duration)
            params.append(ps.period)
            params.append(ps.deadline)
        return params

    def paramsFromList(self, params, solution_old):
        solution = []
        for i in range(0, len(params), 3):
            solution.append(PollingServer(solution_old[i//3].name, params[i], params[i+1], params[i+2], solution_old[i//3].tasks, solution_old[i//3].separation))
        return solution

    # crossover two parents to create two children
    def crossover(self, p1, p2):
        # children are copies of parents by default
        # check for recombination
        if np.random.rand() > self.pCross:
            return p1, p2

        p1_list, p2_list = self.paramsToList(p1), self.paramsToList(p2)

        pt = random.randint(1, len(p1_list))
        # perform crossover
        c1 = p1_list[:pt] + p2_list[pt:]
        c2 = p2_list[:pt] + p1_list[pt:]

        c1, c2 = self.paramsFromList(c1, p1), self.paramsFromList(c2, p2)

        return c1, c2

    # mutation operator
    def mutation(self, solution):
        if np.random.rand() > self.pMut:
            return solution

        s_list = self.paramsToList(solution)
        gene = random.randint(0, len(s_list)-1)

        if gene%3 == 0: #its duration:
            #print("duration", gene, gene%3, s_list)
            s_list[gene] = self.clamp(s_list[gene] + random.randint(-10, 10) * 10, 1, s_list[gene+1])
        elif gene%3 == 1: #its period
            s_list[gene] = self.period_divisors[self.clamp(self.period_divisors.index(s_list[gene]) + random.choice([-1, 0, 1]), 0, len(self.period_divisors)-1)]
        else: #is deadline
            s_list[gene] = self.clamp(s_list[gene] + random.randint(-10, 10) * 10, s_list[gene-2], s_list[gene-1])
        
        new_solution = self.paramsFromList(s_list, solution)

        return new_solution

    # tournament selection
    def selection(self , pop, scores, k=3):
        ind = np.argpartition(-np.array(scores), -10)[-10:]
        return [pop[idx] for idx in ind] 

    def run(self):
        pop = self.initialPopulation()
        # keep track of best solution
        # enumerate generations
        while not self.isTerminationCriteriaMet():
            # evaluate all candidates in the population
            scores =  self.evaluatePopulation(pop)
            # check for new best solution
            newSolution = pop[np.argmin(scores)]
            newCost = np.min(scores)

            if newCost - self.currCost < 0:
                    #print("accepted sol. {}".format(self.isValidNeighbor(newSolution)))
                    self.solution = newSolution
                    self.currCost = newCost

            print(">%d, new best f(%s) = %.3f" % (self.currIter, newSolution, newCost))
            # select parents
            parent_list = self.selection(pop, scores)#[self.selection(pop, scores) for _ in range(self.n_pop)]
            # create the next generation
            children = list()
            parent_list = list(itertools.permutations(parent_list, 2))
            random.shuffle(parent_list)

            for p1, p2 in parent_list[:50]:

                # get selected parents in pairs
                # crossover and mutation
                for c in self.crossover(p1, p2):
                    # mutation
                    self.mutation(c)
                    # store for next generation
                    children.append(c)
            # replace population
            pop = children

            self.currIter += 1


if __name__ == "__main__":
    path = "./test_cases/taskset_small.csv"
    #path = "./test_cases/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()

    """
    optim = Optimizer(TT, ET, 10)
    optim.run()
    optim.printSolution()

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

    schedulable_TT, timetable, wcrt_TT, __ = EDF(TT + sa.solution)
    print(wcrt_TT)
    for ps in sa.solution:
        schedulable_ET, wcrt_ET, __ = EDP(ps)
        print(wcrt_ET)

    sa.printSolution()
    sa.plotCost()

    plotTTtask(TT + sa.solution, timetable, group_tt=True)
    """

    ga = GeneticAlgorithm(TT, ET, 10)
    ga.run()
    ga.printSolution()
    print("all done")