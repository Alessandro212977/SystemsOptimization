
from libraries.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from libraries.algorithms import *

def plotTTtask(TT, sigma):
    plt.rcParams["figure.figsize"] = [20, 7]
    plt.rcParams["figure.autolayout"] = True
    
    fig, gnt = plt.subplots()
    gnt.set_title("Task Scheduling")

    # Setting Y-axis limits
    gnt.set_ylim(0, 27)
    # Setting X-axis limits
    #gnt.set_xlim(0, 27)
    
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Duration')
    gnt.set_ylabel('Tasks')
    
    #plt.xscale("linear")
    #plt.xticks(np.arange(min(x), max(x)-1, 1.0))
    #plt.xticks(np.arange(0, 100, 1.0))
    
    
    y_ticklables = []
    for obj in TT:
        taskname =  "Task" + str(obj.name[3:])
        y_ticklables.append(taskname)
    gnt.set_yticklabels(y_ticklables)
    
    plt.yticks(np.arange(0, len(y_ticklables), 1.0))
        
    res = defaultdict(list)
    for ele in range(len(sigma)):
        if sigma[ele] != "Idle":
            res[sigma[ele]].append(ele)

    cmap = _get_cmap(len(res))
    for key, value in res.items():
        lst = []
        v = 0
        if key < 30:
            while (v < len(value)):
                t = _getDuration(TT, "tTT"+str(key))
                lst.append((value[v] , t))
                v = v + t
            
        gnt.broken_barh(lst, (key, 1), facecolors= cmap(key))
        
    gnt.grid(True)
    plt.savefig("sheduling.png")
    plt.show()

def _getDuration(TT, currtask):
        for obj in TT:
            if obj.name == currtask:
                return obj.duration

def _get_cmap(n, name='hsv'):
        return plt.cm.get_cmap(name, n) 
    
    
            
def plotSimulatedAnealing():
    
    
    pass

if __name__ == "__main__":
    path = "./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = DataLoader(path)
    TT, ET  = dl.loadFile()
    sigma, WCRT = EDF(TT)
    plotTTtask(TT, sigma)
    
        