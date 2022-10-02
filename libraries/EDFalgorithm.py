# load task from file


from ast import main
from math import gcd 
from dataloader import DataLoader as dl

class EDF:
    def __init__(self) -> None:
        pass
    
    def getLcm(tasks):
        temp  = []
        
        for i in range(len(tasks)):
            temp.append(int(tasks[i][2]))
        lcm = temp[0]
        for t in temp[1:]:
            lcm = int(lcm * t/gcd(lcm, t))

        return lcm
        
        

    def edf():
        pass


if __name__ == "__main__":
    tasks = dl("./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv")
    tt = tasks.loadFile()
    
    




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
        