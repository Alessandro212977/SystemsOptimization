# load task from file


from ast import main
from math import gcd, lcm 
import dataloader

class EDF:
    def __init__(self, path) -> None:
        dl = dataloader.DataLoader(path)
        self.TT, self.ET = dl.loadFile()
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
        

    def run(self):
        T = lcm(*[obj.period for obj in self.TT] )
        t=0
        sigma = ["idle"]*T
        R = [0]*len(self.TT)
        WCRT = [0]*len(self.TT)
        C = [obj.duration for obj in self.TT]
        D = [obj.deadline for obj in self.TT]
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
                    print("t:", t, "Im here task:", i, "deadline", task.deadline)

            if all(v == 0 for v in C):
                sigma[t] = "idle"
            else:
                earliest_deadline = min(D_copy)
                deadline_index = D_copy.index(earliest_deadline)
                if(C[deadline_index] > 0):
                    sigma[t] = deadline_index
                    C[deadline_index] -= 1  
                    print("deadline index:", deadline_index, "C", C[deadline_index], D_copy)
                    print(" index of task:", earliest_deadline, D.index(earliest_deadline), D.index(D[deadline_index]), D)
                    print()
                    if (C[deadline_index] == 0):
                        print("reset", C[deadline_index])
                        D_copy[deadline_index] = 100000000   
                else:
                    #print("im inside the else", deadline_index)
                    D_copy[deadline_index] = 100000000
                    earliest_deadline = min(D_copy)
                    deadline_index = D_copy.index(earliest_deadline)
                    sigma[t] = deadline_index
                    C[deadline_index] -= 1
            t += 1
        if all(v > 0 for v in C):
            return "empty"
        
        return sigma, WCRT
if __name__ == "__main__":
    edf = EDF("./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv")
    print(edf.run())

#def load_task():
#     lines = []
#     tasks = []
#     #rawdata = open('asset/tasks.txt', 'r')
#     #rawdata.seek(45)
#     tt_data = open('asset/tt.txt','r')
#     tt_data.seek(10)
#     data = tt_data.readlines()
#     for d in data:
#         lines.append(d.strip('\n'))
#     for line in lines:
#         tasks.append(list(line.split(",")))
    
#     return tasks
        