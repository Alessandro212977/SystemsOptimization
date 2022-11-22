from libraries.algorithms import EDF, EDP
import libraries.dataloader as dataloader
from libraries.tasks import PollingServer
import random
import numpy as np
from sympy import divisors
from math import lcm
import itertools

class geneticAlgorithm:

    def __init__(self, n_pop, n_iter, TTtasks, ETtasks, r_cross=0.8, r_mut=0.8):
        self.n_pop = n_pop
        self.TTtasks = TTtasks
        self.ETtasks = ETtasks
        self.n_iter = n_iter

        self.r_cross = r_cross
        self.r_mut = r_mut

        self.clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

        self.hyperperiod = lcm(*[obj.period for obj in self.TTtasks])
        self.period_divisors = divisors(self.hyperperiod)[1:]

    def initialPopulation(self):
        pop = []
        for __ in range(self.n_pop):
            max_sep = max([task.separation for task in self.ETtasks])
            init_ps = []#[(300, 1000, 1000)]*max_sep   #[(1070, 2000, 1580), (200, 2000, 1470), (100, 1000, 1000)]
            for sep in range(1, max_sep+1):
                tasks = [task for task in self.ETtasks if task.separation==sep]
                period = random.choice(self.period_divisors)
                budget = random.randint(1, period)
                deadline = random.randint(budget, period)
                init_ps.append(PollingServer("PS {}".format(sep), budget, period, deadline, tasks, sep))
            pop.append(init_ps)
        return pop

    def evaluate(self, pop, weights=[0.5, 0.5]):
        assert sum(weights) == 1.0
        costs_list = []
        for solution in pop:
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
            costs_list.append(cost)
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
        if np.random.rand() > self.r_cross:
            return p1, p2

        p1_list, p2_list = self.paramsToList(p1), self.paramsToList(p2)

        pt = random.randint(1, len(p1_list))
        # perform crossover
        #### how to do crossover ?
        c1 = p1_list[:pt] + p2_list[pt:]
        c2 = p2_list[:pt] + p1_list[pt:]

        c1, c2 = self.paramsFromList(c1, p1), self.paramsFromList(c2, p2)

        return c1, c2

    # mutation operator
    def mutation(self, solution):
        if np.random.rand() > self.r_mut:
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
        pop= self.initialPopulation()
        # keep track of best solution
        # enumerate generations
        for gen in range(self.n_iter):
            # evaluate all candidates in the population
            scores =  self.evaluate(pop)
            # check for new best solution
            best = pop[np.argmin(self.evaluate(pop))]
            best_eval = np.min(self.evaluate(pop))

            print(">%d, new best f(%s) = %.3f\n" % (gen, best, best_eval))
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
        return [best, best_eval]



if __name__ == "__main__":
    path = "./test_cases/taskset_small.csv"
    #path = "./test_cases/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()

    ga = geneticAlgorithm(10, 10, TT, ET)
    ga.run()
    print("all done")







