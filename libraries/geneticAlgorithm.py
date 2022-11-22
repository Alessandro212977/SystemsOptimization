from libraries.algorithms import EDF, EDP
import libraries.dataloader as dataloader
from libraries.tasks import PollingServer
import random
import math
import numpy as np
import copy as cp
from sklearn import preprocessing

class geneticAlgorithm:
    def __int__(self, n_pop, n_iter, TTtasks, r_cross, r_mut):
        self.n_pop = n_pop
        self.TTtasks = TTtasks
        self.n_iter = n_iter

    def initialPopulation(self):
        pop = []
        for _ in range(self.n_pop):
            # initial population of random bitstring
            ######check how we are going to pass this over
            indi_ps = PollingServer(np.randint(0, 100), np.randint(0, 1000), np.randint(0, 1000), (ettasks))
            pop.append(indi_ps)
        return pop

    def evaluate(self , ps):
        sigma, wrct = EDF(self.TTtasks + [ps])
        schedule, responsetime = EDP(ps)
        ## how to normilize ?
        score = 0.6 * wrct.mean() + 0.4 * responsetime
        return score

    # crossover two parents to create two children
    def crossover( self,p1, p2, r_cross):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if np.rand() < r_cross:
            # select crossover point
            pt = np.randint(1, 3)
            # perform crossover
            #### how to do crossover ?
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    # mutation operator
    def mutation(self , ps, r_mut):
        for i in range(0, 3):
            # check for a mutation
            if np.rand() < r_mut:
                ## how to mutate
    # tournament selection
    def selection(self , pop, scores, k=3):
        # first random selection
        selection_ix = np.randint(len(pop))
        for ix in np.randint(0, len(pop), k - 1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    def run(self):
        pop= self.initialPopulation()
        # keep track of best solution
        best, best_eval = 0, self.evaluate(pop[0])
        # enumerate generations
        for gen in range(self.n_iter):
            # evaluate all candidates in the population
            scores = [self.evaluate(self , ps) for ps in pop]
            # check for new best solution
            for i in range(self.n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
            # select parents
            selected = [self.selection(pop, scores) for _ in range(self.n_pop)]
            # create the next generation
            children = list()
            for i in range(0, self.n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]
                # crossover and mutation
                for c in self.crossover(p1, p2, self.r_cross):
                    # mutation
                    self.mutation(c, self.r_mut)
                    # store for next generation
                    children.append(c)
            # replace population
            pop = children
        return [best, best_eval]











