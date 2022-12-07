import itertools
import logging
import math
import random
import warnings
from math import lcm

import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
from sympy import divisors
from tqdm import tqdm

import libraries.dataloader as dataloader
import wandb
from libraries.algorithms import EDF, EDP
from libraries.tasks import PollingServer

warnings.filterwarnings("ignore")

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)  # DEBUG


class Optimizer:
    def __init__(
        self,
        TTtasks,
        ETtasks,
        numinstances=1,
        numworkers=1,
        maxiter=100,
        toll=0.1,
        convergence=0.2,
        extra_ps=0,
        wandblogging=False,
    ):
        # Tasks
        self.TTtasks = TTtasks
        self.ETtasks = ETtasks
        self.freeETtasks = [task for task in self.ETtasks if task.separation == 0]  # ET tasks with separation = 0

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
        self.solutions = self.initializeSolutions(self.numInstances, extra_ps)
        self.newSolutions = [None] * self.numInstances
        self.bestSolution = self.solutions[0]  # global

        # Initialize costs
        self.currCosts = self.computeCosts(self.solutions)
        self.newCosts = [None] * self.numInstances
        self.bestCost = self.currCosts[0]  # global

        # Initialize iteration counters
        self.currIter = [0] * self.numInstances

        # Datalog
        self.datalog = [
            {
                "bins": [0],
                "costs": [self.currCosts[idx]],
                "accepted_costs": [self.currCosts[idx]],
                "isvalid": 0,
                "istoll": 0,
                "ismin": 0,
                "is_converged": "NaN",
            }
            for idx in range(self.numInstances)
        ]
        self.convergence = convergence
        self.wandbLog = wandblogging
        if self.wandbLog:
            self.wandbrun = wandb.init(
                project="SystemsOptimization", entity="alessandro26", name=self.__class__.__name__
            )
            # self.wandbrun.config = {"numInstances": self.numInstances,
            #                        "numWorkers": self.numWorkers,
            #                        "maxIter": self.maxIter,
            #                        "toll": self.toll}

    def plotCost(self, instance_idx="best"):
        fig, ax = plt.subplots()

        if instance_idx == "best":
            instance_idx = np.argmin(self.currCosts)
            ax.plot(
                self.datalog[instance_idx]["bins"],
                self.datalog[instance_idx]["accepted_costs"],
                label="Best solution cost",
            )
        elif instance_idx == "all":
            """
            for instance_idx in range(self.numInstances):
                ax.plot(
                    self.datalog[instance_idx]["bins"],
                    self.datalog[instance_idx]["accepted_costs"],
                    label="Instances" if instance_idx == 0 else None,
                    color="gray",
                    linestyle="dashed",
                    linewidth=0.7,
                )
            """
            ax.plot(self.datalog[0]["bins"], self.datalog[0]["mean"], label=f"Mean cost")
            ax.axhline(self.convergence, linewidth=1, color="red", label="threshold")
            if self.numInstances > 1:
                low = self.datalog[0]["mean"] + self.datalog[0]["std"]
                up = np.maximum(
                    self.datalog[0]["mean"] - self.datalog[0]["std"],
                    np.zeros_like(self.datalog[0]["mean"] - self.datalog[0]["std"]),
                )
                ax.fill_between(
                    self.datalog[0]["bins"], low, up, facecolor="blue", alpha=0.3, label=r"$\pm \sigma$ interval"
                )

        # ax.axhline(1, linewidth=1, color="green", label="is_valid")
        #
        ax.set_ylim(bottom=0)
        ax.set_xlim((0, self.maxIter))
        ax.grid()
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Cost")
        ax.set_title("Cost of accepted solutions")
        ax.legend()
        return plt

    def initializeSolutions(self, n=1, num_extra_ps=0):
        solutions = []
        max_sep = max([task.separation for task in self.ETtasks])
        for __ in range(n):
            sol = []
            num_e_ps = random.randint(0, len(self.freeETtasks)) if num_extra_ps == "random" else num_extra_ps
            freeTasksIdx = np.random.choice(range(1, num_e_ps + max_sep + 1), len(self.freeETtasks), replace=True)

            for sep in range(1, max_sep + 1):
                tasks = [task for task in self.ETtasks if task.separation == sep] + [
                    task for i, task in enumerate(self.freeETtasks) if freeTasksIdx[i] == sep
                ]
                period = random.choice(self.period_divisors)
                budget = random.randint(1, period)
                deadline = random.randint(budget, period)
                sol.append(PollingServer("PS {}".format(sep), budget, period, deadline, tasks.copy(), sep))

            for ps_idx in range(max_sep + 1, max_sep + num_e_ps + 1):
                tasks = [task for i, task in enumerate(self.freeETtasks) if freeTasksIdx[i] == ps_idx]

                if len(tasks) > 0:
                    period = random.choice(self.period_divisors)
                    budget = random.randint(1, period)
                    deadline = random.randint(budget, period)
                    sol.append(PollingServer("PS E{}".format(ps_idx), budget, period, deadline, tasks.copy(), 0))

            assert len(self.ETtasks) == sum(
                [len(ps.tasks) for ps in sol]
            ), f"missing some tasks {len(self.ETtasks)} {sum([len(ps.tasks) for ps in sol])}"
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

            cost = 0 if et_schedulable and tt_schedulable else 0.5 * et_penalty + 0.5 * tt_penalty
            for w, c in zip(weights, [np.mean(et_wcrt), np.mean(tt_wcrt)]):
                cost += w * c / self.hyperperiod
            costs.append(cost)

        return costs

    def printSolution(self, get=False):
        string = "-------------------------- Solution --------------------------\n"

        wcrt_tt = EDF(self.TTtasks + self.bestSolution)[2]
        wcrt_et = []
        for ps in self.bestSolution:
            string += str(ps) + "\n"
            wcrt_et = wcrt_et + EDP(ps)[1]

        string += "Optimal solution cost: {:.5f}, valid: {}\n".format(
            self.computeCosts([self.bestSolution])[0],
            self.areValidNeighbors([self.bestSolution])[0],
        )
        string += "Average WCRT for TT+PS task: {:.2f} ms (max: {} ms)\n".format(np.mean(wcrt_tt), max(wcrt_tt))
        string += "Average WCRT for ET task: {:.2f} ms (max: {} ms)\n".format(np.mean(wcrt_et), max(wcrt_et))
        if self.numInstances > 1:
            string += "Stats of {} runs: mean converged (cost<{}) at iter: {}\n".format(
                self.numInstances, self.convergence, self.datalog[0]["is_converged"]
            )
        string += "-------------------------------------------------------------"

        if get:
            return string
        else:
            print(string)

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
            # update iterations
            self.update(idx, pbar)
            self.getNewSolution(idx)

            # if the new solution is better, accept it
            if self.accept(idx):
                self.solutions[idx] = self.newSolutions[idx]
                self.currCosts[idx] = self.newCosts[idx]

            # Log data
            self.datalog[idx]["bins"].append(self.currIter[idx])
            self.datalog[idx]["costs"].append(self.newCosts[idx])
            self.datalog[idx]["accepted_costs"].append(self.currCosts[idx])
            if self.datalog[idx]["accepted_costs"][-1] < 1 and self.datalog[idx]["accepted_costs"][-2] >= 1:
                # entered valid solutions
                self.datalog[idx]["isvalid"] = self.currIter[idx]
            if (
                self.datalog[idx]["accepted_costs"][-1] < self.toll
                and self.datalog[idx]["accepted_costs"][-2] >= self.toll
            ):
                # save iteration when toll is reached
                self.datalog[idx]["istoll"] = self.currIter[idx]
            if self.datalog[idx]["accepted_costs"][-1] < self.datalog[idx]["accepted_costs"][-2]:
                # save iteration of the best cost
                self.datalog[idx]["ismin"] = self.currIter[idx]

            if self.wandbLog:
                # self.wandbrun.log({f"cost_instance_{idx}": self.newCosts[idx], f"iter_instance_{idx}": self.currIter[idx]})
                self.wandbrun.log(
                    {f"accepted_cost_instance_{idx}": self.currCosts[idx], f"iter_instance_{idx}": self.currIter[idx]}
                )

        return idx, self.solutions[idx], self.currCosts[idx], self.currIter[idx], self.datalog[idx]

    def run(self, pbar=None):
        if self.numWorkers == 1:
            for i in range(self.numInstances):
                self.runTask(i, pbar)
        else:
            with mp.Pool(self.numWorkers) as pool:
                for idx, sol, cost, iter, datalog in pool.imap_unordered(
                    self.runTask, [(i) for i in range(self.numInstances)]
                ):
                    self.solutions[idx] = sol
                    self.currCosts[idx] = cost
                    self.currIter[idx] = iter
                    self.datalog[idx] = datalog
                    if pbar:
                        pbar()

        for idx in range(self.numInstances):
            self.datalog[idx]["bins"] = self.datalog[idx]["bins"] + list(
                range(self.datalog[idx]["bins"][-1] + 1, self.maxIter + 1)
            )
            self.datalog[idx]["accepted_costs"] = self.datalog[idx]["accepted_costs"] + [
                self.datalog[idx]["accepted_costs"][-1]
            ] * (self.maxIter - self.currIter[idx])
            self.datalog[idx]["costs"] = self.datalog[idx]["costs"] + [self.datalog[idx]["costs"][-1]] * (
                self.maxIter - self.currIter[idx]
            )

        self.datalog[0]["mean"] = np.mean(
            np.array([self.datalog[idx]["accepted_costs"] for idx in range(self.numInstances)]), axis=0
        )
        std = np.std(np.array([self.datalog[idx]["accepted_costs"] for idx in range(self.numInstances)]), axis=0)
        self.datalog[0]["std"] = std

        self.datalog[0]["is_converged"] = np.argmax(self.datalog[0]["mean"] < self.convergence)

        best_idx = np.argmin(self.currCosts)
        self.bestSolution = self.solutions[best_idx]
        self.bestCost = self.currCosts[best_idx]
        assert self.bestCost == min(self.currCosts)

        if self.wandbLog:
            """
            self.wandbrun.log({"datalog_cost" : wandb.plot.line_series(
                        xs=self.datalog[0]["bins"],
                        ys=[self.datalog[idx]["costs"] for idx in range(self.numInstances)],
                        keys=[f"cost_{idx}" for idx in range(self.numInstances)],
                        title="Costs",
                        xname="Iterations")})
            """
            self.wandbrun.log(
                {
                    "datalog_accepted_cost": wandb.plot.line_series(
                        xs=self.datalog[0]["bins"],
                        ys=[self.datalog[idx]["accepted_costs"] for idx in range(self.numInstances)],
                        keys=[f"cost_{idx}" for idx in range(self.numInstances)],
                        title="Accepted Costs",
                        xname="Iterations",
                    )
                }
            )


class SimulatedAnnealing(Optimizer):
    def __init__(
        self,
        TTtasks,
        ETtasks,
        numinstances=1,
        numworkers=1,
        maxiter=1000,
        toll=0.01,
        convergence=0.2,
        extra_ps="random",
        wandblogging=False,
        iterationPerTemp=100,
        initialTemp=0.1,
        finalTemp=0.0001,
        tempReduction="geometric",
        alpha=0.5,
        beta=5,
        dur_radius=200,
        dln_radius=200,
        priority_prob=0,
        free_tasks_switches=1,
        no_upper_lim=True,
    ):
        super().__init__(TTtasks, ETtasks, numinstances, numworkers, maxiter, toll, convergence, extra_ps, wandblogging)

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

        self.dur_radius = dur_radius
        self.dln_radius = dln_radius
        self.priority_prob = priority_prob
        self.free_tasks_switches = free_tasks_switches
        self.no_upper_lim = no_upper_lim

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
            plt.plot([i * self.currIterationPerTemp for i in iterations], metropolis, marker="x", label=label)
        # inalize plot
        plt.xlabel("Iteration")
        plt.ylabel("Acceptance probability")
        plt.title("Metropolis Criterion")
        # plt.ylim((0.0001, 1))
        # plt.yscale("log")
        plt.grid()
        plt.legend()
        return plt

    def accept(self, idx=0):
        cond = super().accept(idx)
        if self.newCosts[idx] - self.currCosts[idx] > 0:
            # print("the solution could be accepted with prob: {:.3f}".format(math.exp(-cost/self.currTemp)))
            return cond or random.random() < math.exp(-(self.newCosts[idx] - self.currCosts[idx]) / self.currTemps[idx])
        return cond or False

    def switchTasks(self, solution):
        for __ in range(self.free_tasks_switches):
            # choose a ps from where to the the task
            ps_from = random.randint(0, len(solution) - 1)

            # list of possible tasks
            switchable_tasks = [idx for idx, task in enumerate(solution[ps_from].tasks) if task.separation == 0]

            if len(switchable_tasks) > 0:
                # choose a destination ps (If it is a new one, make a new PS)
                ps_to = random.randint(0, len(solution) if self.no_upper_lim else len(solution) - 1)
                if ps_to >= len(solution):
                    new_ps = PollingServer(
                        f"PS E{ps_to}",
                        solution[ps_from].duration,
                        solution[ps_from].period,
                        solution[ps_from].deadline,
                        tasks=[],
                        separation=0,
                    )
                    solution.append(new_ps)

                # choose a task and do the switch
                solution[ps_to].tasks.append(solution[ps_from].tasks.pop(random.choice(switchable_tasks)))

                # if the from_ps is now empty then remove it
                if len(solution[ps_from].tasks) == 0:
                    solution.pop(ps_from)
        return solution

    def switchPriorities(self, solution):
        for ps in solution:
            for task in ps.tasks:
                if random.random() < self.priority_prob:
                    task.priority = self.clamp(task.priority + random.randint(-1, 1), 0, 6)

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

            duration = self.clamp(ps.duration, 1, new_period)
            dur_low, dur_up = max(1, duration - self.dur_radius), min(new_period, duration + self.dur_radius)
            new_duration = random.randint(dur_low, dur_up)

            deadline = self.clamp(ps.deadline, new_duration, new_period)
            dln_low, dln_up = max(new_duration, deadline - self.dln_radius), min(new_period, deadline + self.dln_radius)
            new_deadline = random.randint(dln_low, dln_up)

            new_center.append(
                PollingServer(
                    ps.name,
                    new_duration,
                    new_period,
                    new_deadline,
                    ps.tasks.copy(),
                    ps.separation,
                )
            )

        # switch ET tasks with sep = 0
        new_center = self.switchTasks(new_center)

        self.switchPriorities(new_center)

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
        convergence=0.2,
        wandblogging=False,
        pop_size=20,
        num_parents=8,
        p_cross=0.9,
        p_mut=0.1,
        selection="rank",
        free_tasks_switches=1,
    ):
        super().__init__(
            TTtasks,
            ETtasks,
            numinstances,
            numworkers,
            maxiter,
            toll,
            convergence,
            extra_ps=0,
            wandblogging=wandblogging,
        )

        self.popSize = pop_size
        self.numParents = num_parents
        self.pCross = p_cross
        self.pMut = p_mut
        self.selectionMode = selection
        self.free_tasks_switches = free_tasks_switches

        self.populations = self.initialPopulations()
        self.scores = [self.computeCosts(self.populations[idx]) for idx in range(self.numInstances)]

    def initialPopulations(self):
        def func(args):
            return self.initializeSolutions(self.popSize, num_extra_ps=0)

        pop = []
        if self.numWorkers == 1:
            for __ in range(self.numInstances):
                pop.append(self.initializeSolutions(self.popSize, num_extra_ps=0))
            return pop
        else:
            with mp.Pool(self.numWorkers) as pool:
                for single_pop in pool.imap_unordered(func, [idx for idx in range(self.numInstances)]):
                    pop.append(single_pop)
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
                    solution_old[i // 3].tasks.copy(),
                    solution_old[i // 3].separation,
                )
            )
        return solution

    def crossover(self, p1, p2):
        # crossover two parents to create two children
        if random.random() > self.pCross:
            return p1, p2

        pt = random.randint(1, len(p1) - 1)
        c1, c2 = p1.copy(), p2.copy()
        for i, (ps1, ps2) in enumerate(zip(p1, p2)):
            if i<pt:
                c1[i].duration = ps2.duration
                c1[i].period = ps2.period
                c1[i].deadline = ps2.deadline
                c2[i].duration = ps1.duration
                c2[i].period = ps1.period
                c2[i].deadline = ps1.deadline
        return c1, c2

    def tasksMutation(self, solution):
        for __ in range(self.free_tasks_switches):
            # choose a ps from where to the the task
            ps_from = random.randint(0, len(solution) - 1)
            # list of possible tasks
            switchable_tasks = [idx for idx, task in enumerate(solution[ps_from].tasks) if task.separation == 0]

            if len(switchable_tasks) > 0:
                # choose a destination ps (If it is a new one, make a new PS)
                ps_to = random.randint(0, len(solution) - 1)
                # choose a task and do the switch
                solution[ps_to].tasks.append(solution[ps_from].tasks.pop(random.choice(switchable_tasks)))
        return solution

    def mutation(self, solution):
        # mutation operator
        if random.random() > self.pMut:
            return solution

        new_solution = solution.copy()
        ps_idx = random.randint(0, len(solution) - 1)
        ps =  new_solution[ps_idx]
        new_period = self.period_divisors[
            self.clamp(
                self.period_divisors.index(ps.period) + random.choice([-1, 0, 1]),
                0,
                len(self.period_divisors) - 1,
            )
        ]

        duration = self.clamp(ps.duration, 1, new_period)
        dur_low, dur_up = max(1, duration - 200), min(new_period, duration + 200)
        new_duration = random.randint(dur_low, dur_up)

        deadline = self.clamp(ps.deadline, new_duration, new_period)
        dln_low, dln_up = max(new_duration, deadline - 200), min(new_period, deadline + 200)
        new_deadline = random.randint(dln_low, dln_up)

        new_solution[ps_idx] = PollingServer(
                ps.name,
                new_duration,
                new_period,
                new_deadline,
                ps.tasks.copy(),
                ps.separation,
            )

        # switch tasks with separation 0
        new_solution = self.tasksMutation(new_solution)

        return new_solution

    # random selection
    def selection(self, pop, scores):
        if self.selectionMode == "rank":
            ind = np.argpartition(-np.array(scores), -self.numParents)[-self.numParents :]
            return [pop[i] for i in ind]

        elif self.selectionMode == "tournament":  # how is it called?
            parents = []
            for __ in range(0, self.numParents):
                selected_parent = random.randint(0, self.popSize - 1)
                for __ in range(0, 5):
                    k = random.randint(0, self.popSize - 1)
                    # check if better
                    if scores[k] < scores[selected_parent]:
                        selected_parent = k
                parents.append(pop[selected_parent])
            return parents

        else:
            raise NotImplementedError(f"Required selection mode ({self.selectionMode}) is not available")

    def getNewSolution(self, idx):
        parent_list = self.selection(self.populations[idx], self.scores[idx])

        # create the next generation
        children = []
        parent_list = list(itertools.permutations(parent_list, 2))
        random.shuffle(parent_list)

        for p1, p2 in parent_list[: self.popSize // 2]:
            # crossover
            c1, c2 = self.crossover(p1, p2)
            # mutation
            c1, c2 = self.mutation(c1), self.mutation(c2)

            # store for next generation
            children.append(c1)
            children.append(c2)

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
        import cProfile
        import pstats

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
