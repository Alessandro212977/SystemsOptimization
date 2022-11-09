
from libraries.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from libraries.algorithms import *
from numpy import arange
    
def plotTTtask(TT, sigma, xmax=None):
    plt.rcParams["figure.figsize"] = [20, 7]
    plt.rcParams["figure.autolayout"] = True
    
    fig, gnt = plt.subplots()
    gnt.set_title("Task Scheduling")

    #X-axis
    if xmax:
        gnt.set_xlim(0, xmax)
        plt.xticks(np.arange(0, xmax, xmax//30))
        gnt.set_xlabel('Duration')
        
    #Y-axis
    y_ticklables = [obj.name for obj in TT]
    plt.yticks(np.arange(0, len(y_ticklables), 1.0))
    gnt.set_yticklabels(y_ticklables, va="bottom")
    gnt.set_ylim(0, len(y_ticklables))
    gnt.set_ylabel('Tasks')
        
    #data
    res = defaultdict(list)
    for i, value in enumerate(sigma):
        if value != "Idle":
            res[value].append(i)

    cmap = plt.cm.get_cmap('hsv', len(res))
    for key, value in res.items():
        lst = []
        v = 0
        while (v < len(value)):
            t = {i: obj.duration for i, obj in enumerate(TT)}[key]
            lst.append((value[v] , t))
            v += t
        gnt.broken_barh(lst, (key, 1), facecolors= cmap(key))
        
    gnt.grid(True)
    plt.savefig("sheduling.png")
    plt.show()
     
def plotSimulatedAnealing():
    
    # objective function
    
    # define range for input
    r_min, r_max = -5.0, 5.0
    # sample input range uniformly at 0.1 increments
    inputs = arange(r_min, r_max, 0.1)
    # compute targets
    results = [_objective([x]) for x in inputs]
    # create a line plot of input vs result
    plt.plot(inputs, results)
    # define optimal input value
    x_optima = 0.0
    # draw a vertical line at the optimal input
    plt.axvline(x=x_optima, ls='--', color='red')
    # show the plot
    plt.show()

def _objective(x):
        return x[0]**2.0

def run(name):
    path = "./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = DataLoader(path)
    
    if name == "TT":
        TT, ET  = dl.loadFile()
        sigma, WCRT = EDF(TT)
        plotTTtask(TT, sigma)
    else:
        plotSimulatedAnealing()
        
    
if __name__ == "__main__":
    path = "./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = DataLoader(path)
    TT, ET  = dl.loadFile()
    ps = PollingServer("ps", duration=18, period=2000, deadline=2000, tasks=ET)
    sigma, WCRT = EDF(TT+[ps])
    #print(sigma, WCRT)
    plotTTtask(TT+[ps], sigma)
    #plotSimulatedAnealing()
    
        