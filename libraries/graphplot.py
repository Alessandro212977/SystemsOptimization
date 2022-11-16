
from libraries.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from libraries.algorithms import *
from numpy import arange
    
def plotTTtask(TT, sigma, xmax=None):
    plt.rcParams["figure.figsize"] = [16, 7]
    plt.rcParams["figure.autolayout"] = True
    
    fig, ax = plt.subplots()
    ax.set_title("Task Scheduling")

    #X-axis
    if not xmax:
        xmax = lcm(*[obj.period for obj in TT])
    ax.set_xlim(0-0.01*xmax, xmax+0.01*xmax)
    plt.xticks(np.arange(0, xmax+xmax//(xmax/1000), xmax//(xmax/1000)))
    ax.set_xlabel('Duration')
        
    #Y-axis
    y_ticklables = [obj.name for obj in TT]
    plt.yticks(np.arange(0, len(y_ticklables), 1.0))
    ax.set_yticklabels(y_ticklables, va="bottom")
    ax.set_ylim(0, len(y_ticklables))
    ax.set_ylabel('Tasks')
        
    #data
    res = defaultdict(list)
    current_val = sigma[0]
    starting_idx = 0
    length = 1
    for i, val in enumerate(sigma[1:]):
        if val == current_val:
            length += 1
        else:
            res[current_val].append((starting_idx, length))
            length = 1
            starting_idx = i+1
            current_val = val
    res[current_val].append((starting_idx, length))

    cmap = plt.cm.get_cmap('hsv', len(res))
    for key, value in res.items():
        if key=="idle" or key=="Idle":
            continue
        ax.broken_barh(value, (key, 1), facecolors=cmap(key))
        
    ax.grid(True)
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
    path = "./test_cases/taskset_small.csv"
    dl = DataLoader(path)
    TT, ET  = dl.loadFile()
    ps = PollingServer("ps", duration=1000, period=2000, deadline=1000, tasks=ET, separation=0)
    schedulable, sigma, WCRT = EDF(TT+[ps])
    #print(sigma, WCRT)
    plotTTtask(TT+[ps], sigma)
    #plotSimulatedAnealing()
    
        