import itertools
import random
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from sympy import divisors
from math import lcm
import multiprocess as mp
from tqdm import tqdm
import wandb

from libraries.algorithms import EDF, EDP
import libraries.dataloader as dataloader
from libraries.tasks import PollingServer
from libraries.graphplot import getTimetablePlot

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)  # DEBUG


class Optimizer:
    def __init__(self, TTtasks, ETtasks, numinstances=1, numworkers=1, maxiter=100, toll=0.1, wandblogging=True):
        # Tasks
        self.TTtasks = TTtasks
        self.ETtasks = ETtasks

        # Instances and workers
        self.numInstances = numinstances
        self.numWorkers = numworkers
        assert self.numInstances >= self.numWorkers, "More workers than instances"

        # Termination Criteria
        self.maxIter = maxiter
        self.toll = toll

        # Periods
        self.hyperperiod = lcm(*[obj.period for obj in self.TTtasks])
        self.period_divisors = divisors(self.hyperperiod)[1:]

        # Utilities
        self.clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

        # Initialize solutions
        self.solutions = self.initializeSolutions(self.numInstances)
        self.newSolutions = [None] * self.numInstances
        self.bestSolution = self.solutions[0]  # global

        # Initialize costs
        self.currCosts = self.computeCosts(self.solutions)
        self.newCosts = [None] * self.numInstances
        self.bestCost = self.currCosts[0]  # global

        # Initialize iteration counters
        self.currIter = [0] * self.numInstances

        # Datalog
        self.datalog = [{"bins": [], "costs": [], "accepted_costs": [], "isvalid": 0, "ismin": 0} for __ in range(self.numInstances)]
        self.wandbLog = wandblogging
        if self.wandbLog:
            self.wandbrun = wandb.init(project="SystemsOptimization", entity="alessandro26", name=self.__class__.__name__)
            #self.wandbrun.config = {"numInstances": self.numInstances,
            #                        "numWorkers": self.numWorkers,
            #                        "maxIter": self.maxIter,
            #                        "toll": self.toll}

    def plotCost(self, instance_idx='best'):
        fig, ax = plt.subplots()

        if instance_idx == 'best':
            instance_idx = np.argmin(self.currCosts)
            ax.plot(self.datalog[instance_idx]["bins"], self.datalog[instance_idx]["accepted_costs"], label="Best solution cost")
        elif instance_idx == 'all':
            for instance_idx in range(self.numInstances):
                ax.plot(self.datalog[instance_idx]["bins"], self.datalog[instance_idx]["accepted_costs"], label=f"Instance {instance_idx}", color="#1f77b4", linestyle="dashed", linewidth=0.7)
            ax.plot(self.datalog[0]["bins"], self.datalog[0]["mean"], label=f"Mean cost", color="red")

        ax.set_ylim((0, 3))
        ax.set_xlim((0, self.maxIter))
        ax.grid()
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        ax.set_title("Cost of accepted solutions")
        plt.show()

    def initializeSolutions(self, n=1):
        solutions = []
        for __ in range(n):
            max_sep = max([task.separation for task in self.ETtasks])
            sol = []
            for sep in range(1, max_sep + 1):
                tasks = [task for task in self.ETtasks if task.separation == sep]
                period = random.choice(self.period_divisors)
                budget = random.randint(1, period)
                deadline = random.randint(budget, period)
                sol.append(PollingServer("PS {}".format(sep), budget, period, deadline, tasks, sep))
            solutions.append(sol)
        return solutions

    def areSatisfyingConstraints(self, solutions: list) -> list:
        result = []
        for sol in solutions:
            cond = True
            for ps in sol:
                cond = cond and ps.duration <= ps.period and ps.duration <= ps.deadline <= ps.period
            result.append(cond)
        return result

    def areValidNeighbors(self, solutions: list) -> list:
        results = self.areSatisfyingConstraints(solutions)
        for i, sol in enumerate(solutions):
            for ps in sol:
                results[i] = results[i] and EDP(ps)[0] and EDF(self.TTtasks + sol)[0]
        return results

    def computeCosts(self, solutions, weights=[0.5, 0.5]):
        assert sum(weights) == 1.0

        costs = []
        for sol in solutions:
            # EDP
            et_schedulable, et_wcrt, et_penalty = True, [], 0
            for ps in sol:
                schedulable, WCRT, penalty = EDP(ps)
                et_schedulable = et_schedulable and schedulable
                et_wcrt = et_wcrt + WCRT
                et_penalty += penalty
            et_penalty /= len(sol)

            # EDF
            tt_schedulable, __, tt_wcrt, tt_penalty = EDF(self.TTtasks + sol)

            cost = 0 if et_schedulable and tt_schedulable else 1 + 0.5 * et_penalty + 0.5 * tt_penalty
            for w, c in zip(weights, [np.mean(et_wcrt), np.mean(tt_wcrt)]):
                cost += w * c / self.hyperperiod
            costs.append(cost)

        return costs

    def printSolution(self):
        print("-------------------------- Solution --------------------------")
        wcrt_tt = EDF(self.TTtasks + self.bestSolution)[2]
        wcrt_et = []
        for ps in self.bestSolution:
            print(ps)
            wcrt_et = wcrt_et + EDP(ps)[1]
        print(
            "Solution cost: {:.5f}, valid: {}".format(
                self.computeCosts([self.bestSolution])[0],
                self.areValidNeighbors([self.bestSolution])[0],
            )
        )
        print("Average WCRT for TT+PS task: {:.2f} ms (max: {} ms)".format(np.mean(wcrt_tt), max(wcrt_tt)))
        print("Average WCRT for ET task: {:.2f} ms (max: {} ms)".format(np.mean(wcrt_et), max(wcrt_et)))
        print("-------------------------------------------------------------")

    def isTerminationCriteriaMet(self, idx=0) -> bool:
        # Termination criteria
        return self.currIter[idx] >= self.maxIter or self.currCosts[idx] < self.toll

    def accept(self, idx=0) -> bool:
        return self.newCosts[idx] - self.currCosts[idx] < 0

    def update(self, idx=0, pbar=None):
        self.currIter[idx] += 1
        if pbar:
            pbar()

    def getNewSolution(self, idx=0):
        self.newSolutions[idx] = self.solutions[idx]
        self.newCosts[idx] = self.computeCosts([self.solutions[idx]])[0]

    def runTask(self, idx, pbar=None):
        while not self.isTerminationCriteriaMet(idx):
            self.getNewSolution(idx)
            
            # if the new solution is better, accept it
            if self.accept(idx):
                self.solutions[idx] = self.newSolutions[idx]
                self.currCosts[idx] = self.newCosts[idx]

            # Log data
            self.datalog[idx]["bins"].append(self.currIter[idx])
            self.datalog[idx]["costs"].append(self.newCosts[idx])
            self.datalog[idx]["accepted_costs"].append(self.currCosts[idx])
            if self.wandbLog:
                #self.wandbrun.log({f"cost_instance_{idx}": self.newCosts[idx], f"iter_instance_{idx}": self.currIter[idx]})
                self.wandbrun.log({f"accepted_cost_instance_{idx}": self.currCosts[idx], f"iter_instance_{idx}": self.currIter[idx]})
                                    
            self.update(idx, pbar)
        return idx, self.solutions[idx], self.currCosts[idx], self.datalog[idx]

    def run(self, pbar=None):
        if self.numWorkers == 1:
            for i in range(self.numInstances):
                self.runTask(i, pbar)
        else:
            with mp.Pool(self.numWorkers) as pool:
                for idx, sol, cost, datalog in pool.imap_unordered(self.runTask, [(i) for i in range(self.numInstances)]):
                    self.solutions[idx] = sol
                    self.currCosts[idx] = cost
                    self.datalog[idx] = datalog
                    if pbar:
                        pbar()

        self.datalog[0]["mean"] = [np.mean([self.datalog[idx]["accepted_costs"][i] for idx in range(self.numInstances)]) for i in range(self.maxIter)]
        self.bestSolution = self.solutions[np.argmin(self.currCosts)]
        self.bestCost = min(self.currCosts)
        if self.wandbLog:
            """
            self.wandbrun.log({"datalog_cost" : wandb.plot.line_series(
                        xs=self.datalog[0]["bins"], 
                        ys=[self.datalog[idx]["costs"] for idx in range(self.numInstances)],
                        keys=[f"cost_{idx}" for idx in range(self.numInstances)],
                        title="Costs",
                        xname="Iterations")})
            """
            self.wandbrun.log({"datalog_accepted_cost" : wandb.plot.line_series(
                        xs=self.datalog[0]["bins"], 
                        ys=[self.datalog[idx]["accepted_costs"] for idx in range(self.numInstances)],
                        keys=[f"cost_{idx}" for idx in range(self.numInstances)],
                        title="Accepted Costs",
                        xname="Iterations")})



class SimulatedAnnealing(Optimizer):
    def __init__(
        self,
        TTtasks,
        ETtasks,
        numinstances=1,
        numworkers=1,
        maxiter=1000,
        toll=0.01,
        wandblogging=True,
        iterationPerTemp=100,
        initialTemp=0.1,
        finalTemp=0.0001,
        tempReduction="geometric",
        alpha=0.5,
        beta=5,
    ):
        super().__init__(TTtasks, ETtasks, numinstances, numworkers, maxiter, toll, wandblogging)

        self.currTemps = [initialTemp] * self.numInstances
        self.finalTemp = finalTemp
        self.currIterationPerTemp = iterationPerTemp
        self.alpha = alpha
        self.beta = beta

        self.decrementRule = {
            "linear": self.linearTempReduction,  # t = t - a
            "geometric": self.geometricTempReduction,  # t = t * a
            "slowDecrease": self.slowDecreaseTempReduction,  # t = t / 1 + Bt
        }[tempReduction]

    def linearTempReduction(self, idx):
        self.currTemps[idx] -= self.alpha

    def geometricTempReduction(self, idx):
        self.currTemps[idx] *= self.alpha

    def slowDecreaseTempReduction(self, idx):
        self.currTemps[idx] = self.currTemps[idx] / (1 + self.beta * self.currTemps[idx])

    def plotTemperature(self):
        iterations = list(range(int(self.maxIter / self.currIterationPerTemp) + 1))
        temperatures = [self.currTemps[0] * self.alpha**i for i in iterations]
        differences = [0.001, 0.01, 0.1, 1.0]
        for d in differences:
            metropolis = [np.exp(-d / t) for t in temperatures]
            # plot iterations vs metropolis
            label = "diff=%.3f" % d
            plt.plot(iterations, metropolis, marker="x", label=label)
        # inalize plot
        plt.xlabel("Iteration")
        plt.ylabel("Metropolis Criterion")
        # plt.ylim((0.0001, 1))
        # plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.show()

    def accept(self, idx=0):
        cond = super().accept(idx)
        if self.newCosts[idx] - self.currCosts[idx] > 0:
            # print("the solution could be accepted with prob: {:.3f}".format(math.exp(-cost/self.currTemp)))
            return cond or random.random() < math.exp(-(self.newCosts[idx] - self.currCosts[idx]) / self.currTemps[idx])
        return cond or False

    def getNewSolution(self, idx):
        new_center = []
        for ps in self.solutions[idx]:
            new_period = self.period_divisors[
                self.clamp(
                    self.period_divisors.index(ps.period) + random.choice([-1, 0, 1]),
                    0,
                    len(self.period_divisors) - 1,
                )
            ]
            new_duration = self.clamp(ps.duration + random.randint(-10, 10) * 10, 1, new_period)
            new_deadline = self.clamp(ps.deadline + random.randint(-10, 10) * 10, new_duration, new_period)
            new_center.append(
                PollingServer(
                    ps.name,
                    new_duration,
                    new_period,
                    new_deadline,
                    ps.tasks,
                    ps.separation,
                )
            )
        self.newSolutions[idx] = new_center
        self.newCosts[idx] = self.computeCosts([new_center])[0]

    def update(self, idx=0, pbar=None):
        super().update(idx, pbar)
        if self.currIter[idx] % self.currIterationPerTemp == 0:
            self.decrementRule(idx)


class GeneticAlgorithm(Optimizer):
    def __init__(
        self,
        TTtasks,
        ETtasks,
        numinstances=1,
        numworkers=1,
        maxiter=10,
        toll=0.01,
        wandblogging=True,
        pop_size=20,
        num_parents=8,
        p_cross=0.9,
        p_mut=0.1,
    ):
        super().__init__(TTtasks, ETtasks, numinstances, numworkers, maxiter, toll, wandblogging)

        self.popSize = pop_size
        self.numParents = num_parents
        self.pCross = p_cross
        self.pMut = p_mut

        self.populations = self.initialPopulations()
        self.scores = [self.computeCosts(self.populations[idx]) for idx in range(self.numInstances)]

    def initialPopulations(self):
        pop = []
        for __ in range(self.numInstances):
            pop.append(self.initializeSolutions(self.popSize))
        return pop

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
            solution.append(
                PollingServer(
                    solution_old[i // 3].name,
                    params[i],
                    params[i + 1],
                    params[i + 2],
                    solution_old[i // 3].tasks,
                    solution_old[i // 3].separation,
                )
            )
        return solution

    def crossover(self, p1, p2):
        # crossover two parents to create two children
        if np.random.rand() > self.pCross:
            return p1, p2

        p1_list, p2_list = self.paramsToList(p1), self.paramsToList(p2)

        pt = random.randint(1, len(p1_list))
        # perform crossover
        c1 = p1_list[:pt] + p2_list[pt:]
        c2 = p2_list[:pt] + p1_list[pt:]

        c1, c2 = self.paramsFromList(c1, p1), self.paramsFromList(c2, p2)

        return c1, c2

    def mutation(self, solution):
        # mutation operator
        if np.random.rand() > self.pMut:
            return solution

        s_list = self.paramsToList(solution)
        gene = random.randint(0, len(s_list) - 1)

        if gene % 3 == 0:  # its duration:
            # print("duration", gene, gene%3, s_list)
            s_list[gene] = self.clamp(s_list[gene] + random.randint(-10, 10) * 10, 1, s_list[gene + 1])
        elif gene % 3 == 1:  # its period
            s_list[gene] = self.period_divisors[
                self.clamp(
                    self.period_divisors.index(s_list[gene]) + random.choice([-1, 0, 1]),
                    0,
                    len(self.period_divisors) - 1,
                )
            ]
        else:  # is deadline
            s_list[gene] = self.clamp(
                s_list[gene] + random.randint(-10, 10) * 10,
                s_list[gene - 2],
                s_list[gene - 1],
            )

        new_solution = self.paramsFromList(s_list, solution)

        return new_solution

    def selection(self, pop, scores):
        # tournament selection
        ind = np.argpartition(-np.array(scores), -self.numParents)[-self.numParents :]
        return [pop[idx] for idx in ind]

    def getNewSolution(self, idx):
        parent_list = self.selection(self.populations[idx], self.scores[idx])

        # create the next generation
        children = []
        parent_list = list(itertools.permutations(parent_list, 2))
        random.shuffle(parent_list)

        for p1, p2 in parent_list[: self.popSize // 2]:
            # crossover and mutation
            for c in self.crossover(p1, p2):
                # mutation
                self.mutation(c)
                # store for next generation
                children.append(c)
        # replace population
        self.populations[idx] = children
        self.scores[idx] = self.computeCosts(children)

        best = np.argmin(self.scores[idx])
        self.newSolutions[idx] = self.populations[idx][best]
        self.newCosts[idx] = self.scores[idx][best]


if __name__ == "__main__":
    path = "./test_cases/taskset_small.csv"
    # path = "./test_cases/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()

    optim = Optimizer(TT, ET, numinstances=2, numworkers=1, maxiter=10)
    optim.run()
    optim.printSolution()

    sa = SimulatedAnnealing(TT, ET, numinstances=2, numworkers=2, maxiter=10)
    # sa.plotTemperature()

    if False:
        import cProfile, pstats

        with cProfile.Profile() as pr:
            with tqdm(total=sa.maxIter, desc="Iterations") as bar:  # progress bar
                sa.run(bar.update)
        pr = pstats.Stats(pr)
        pr.sort_stats("cumulative").print_stats(10)
    else:
        with tqdm(total=sa.maxIter, desc="Iterations") as bar:
            sa.run(bar.update)

    sa.printSolution()
    # sa.plotCost(0)

    # getTimetablePlot(TT + sa.bestSolution, timetable, group_tt=True).show()

    ga = GeneticAlgorithm(TT, ET, numinstances=2, numworkers=2, num_parents=4, pop_size=10, maxiter=3)
    ga.run()
    ga.printSolution()
    print("all done")
