# load task from file

from math import gcd, lcm
from telnetlib import theNULL

from matplotlib.cbook import index_of
import dataloader
from pollingserver import PollingServer as ps
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter


import matplotlib.ticker as ticker

class EDF:
    def __init__(self, path) -> None:
        dl = dataloader.DataLoader(path)
        self.TT, self.ET = dl.loadFile()
        self.poll_server = ps(5, 2000, 5000, self.ET, "polling_server")
        self.TT.append(self.poll_server)
        #self.TT = self.TT[:5]
        #self.ET = self.ET[:5]
    
    def getLcm(tasks):
        temp  = []
        
        for i in range(len(tasks)):
            temp.append(int(tasks[i][2]))
        lcm = temp[0]
        for t in temp[1:]:
            lcm = int(lcm * t/gcd(lcm, t))

        return lcm

    def schedulepollingserver(self):
        
        pass
    
    def getDuration(self, task):
        for obj in self.TT:
            if obj.name == task:
                return obj.duration

    def run(self):
        # LCM of TT task periods
        T = lcm(*[obj.period for obj in self.TT] )
        t=0
        sigma = ["idle"]*T
        R = [0]*len(self.TT)
        WCRT = [0]*len(self.TT)
        C = [obj.duration for obj in self.TT]
        D = [obj.deadline for obj in self.TT]
        N = [obj.name for obj in self.TT]
        D_copy = [obj.deadline for obj in self.TT]
        while t < T:
            for i, task in enumerate(self.TT):
                if C[i] > 0 and D[i] <= t:
                    print("t:", t, "deadline:", D[i], "C[i]:", C[i])
                    return "Empty"
                if C[i] == 0 and D[i] >= t:
                    if t-R[i] >= WCRT[i]:
                        WCRT[i] = t - R[i]
                        #print("WCRT:", WCRT[i])

                if t%task.period == 0:
                    R[i] = t
                    C[i] = task.duration 
                    D[i] = t + task.deadline
                    D_copy[i] = t + task.deadline
                    
                    #print("t:", t, "Im here task:", i, "deadline", task.deadline)
                    
            if all(v == 0 for v in C):
                sigma[t] = "Idle"
            else:
                earliest_deadline = min(D_copy)
                deadline_index = D_copy.index(earliest_deadline)
                if(C[deadline_index] > 0):
                    if (N[deadline_index] == 'polling_server'):
                        self.poll_server.getTask()
                    sigma[t] = deadline_index
                    C[deadline_index] -= 1  
                    #print("deadline index:", deadline_index, "C", C[deadline_index], D_copy)
                    #print(" index of task:", earliest_deadline, D.index(earliest_deadline), D.index(D[deadline_index]), D)
                    #print()
                    # if C[deadline_index] == 0 and D[deadline_index] >= t:
                    #     #print("reset", C[deadline_index])
                    #     if t - R[deadline_index] >= WCRT[deadline_index]:
                    #         WCRT[deadline_index] = t - R[deadline_index]
                        #D_copy[deadline_index] = 100000000   
                else:
                    #print("im inside the else", deadline_index)
                    D_copy[deadline_index] = 100000000
                    earliest_deadline = min(D_copy)
                    deadline_index = D_copy.index(earliest_deadline)
                    sigma[t] = deadline_index
                    C[deadline_index] -= 1
            t += 1
            
        if all(v > 0 for v in C):
            return "Empty"
        
        return sigma, WCRT
if __name__ == "__main__":
    edf = EDF("./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv")
    print(edf.run())
    
    lst = edf.run()
    file1 = open("MyFile.txt", "w") 
    for line in lst:
        file1.writelines(str(line))
        
    x = [1, 1, 1, 1, 6, 6, 6,1 ,1 , 1,1 ]
    
    book = xlsxwriter.Book('test.xlsx')     
    sheet = book.add_sheet()   
    row = 0
    column = 0
    
    for i in x:
        sheet.write(row, column, i)
        
        row += 1
    
    
    

    # df = any
    # for i in x:
    #     duration = edf.getDuration("tTT"+str(i))
    #     plt.figure(figsize=(8,4))
    #     plt.barh(y="Task"+str(i), left = 0, width=duration)
    #     left = duration - 1
        
    #     # plt.show()
        
    
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, gnt = plt.subplots()
 
    # Setting Y-axis limits
    gnt.set_ylim(0, 27)
    # Setting X-axis limits
    gnt.set_xlim(0, 27)
    
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('Duration')
    gnt.set_ylabel('Tasks')
    
    plt.xscale("linear")
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    #gnt.set_yticks([0, 1, 2])
    
    
    y_ticklables = []
    
    for obj in edf.TT:
        taskname =  "Task" + str(obj.name[-1])
        y_ticklables.append(taskname)
    gnt.set_yticklabels(y_ticklables)
    
    print(len(y_ticklables))
    
    # x_ticklables = []
    # for i in range(0, 100):
    #     x_ticklables.append(i)
        
    # gnt.set_xticklabels(x_ticklables)
    

    # Declaring a bar in schedule
    
    
   

        
    h = 10 
    lst = []
    start = 0
    for i in x:
        
        lst.append((start, 4))
        gnt.broken_barh(lst, (h, 5), facecolors='tab:blue')
        start = start + 4
        h = h + 10
          
    
    # gnt.broken_barh([(110, 30), (150, 10)], (50, 9), facecolors='tab:blue')
    # gnt.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9), facecolors='tab:orange')
    
    
    plt.show()
    
    
    
    #gnt.grid(True)
    # Setting ticks on y-axis
    #gnt.set_yticks([15, 25, 35])
    # Labelling tickes of y-axis
    
   
    
    
    # Setting graph attribute
    
     #Declaring multiple bars in at same level and same width
    #gnt.broken_barh([(110, 10), (150, 10)], (10, 9),facecolors ='tab:blue')
    
    #gnt.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9), facecolors =('tab:red'))
    
    # plt.savefig("gantt1.png")