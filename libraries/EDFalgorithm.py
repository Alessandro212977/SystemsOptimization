# load task from file

from math import gcd, lcm 
import dataloader

class EDF:
    def __init__(self, path) -> None:
        dl = dataloader.DataLoader(path)
        self.TT, self.ET = dl.loadFile()
    
    def getLcm(tasks):
        temp  = []
        
        for i in range(len(tasks)):
            temp.append(int(tasks[i][2]))
        lcm = temp[0]
        for t in temp[1:]:
            lcm = int(lcm * t/gcd(lcm, t))

        return lcm
        

    def run(self):
        # LCM of TT task periods
        T = lcm(*[obj.period for obj in self.TT] )
        t=0
        sigma = [0]*T
        R = [0]*len(self.TT)
        WCRT = [0]*len(self.TT)
        C = [obj.duration for obj in self.TT]
        D = [obj.deadline for obj in self.TT]
        while t < T:   
            for i, task in enumerate(self.TT):
                if C[i] > 0 and D[i] <= t:
                    print("t:", t, "deadline:", D[i], "C[i]:", C[i])
                    return "Empty"
                if C[i] == 0 and D[i] >= t:
                    if t-R[i] >= WCRT[i]:
                        WCRT[i] = t - R[i]
                        #print("WCRT:", WCRT[i])
               
                if t % task.period == 0:
                    R[i] = t
                    C[i] = task.duration 
                    D[i] = t + task.deadline
                    #print(R[i], C[i], D[i])
                    print("t:", t, "Im here")

            if all(v == 0 for v in C):
                sigma[t] = "Idle"
            else:
                sigma[t] = task
                C[i] -= 1     
            t += 1
            
        if all(v > 0 for v in C):
            return "Empty"
        
        return sigma, WCRT
if __name__ == "__main__":
    edf = EDF("./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv")
    print(edf.run())
<<<<<<< HEAD

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
        
=======
>>>>>>> b308003064822be2989acb384c8ecc1f7e245683
