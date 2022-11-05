from libraries.tasks import PollingServer
import libraries.dataloader as dataloader
from math import lcm, ceil

def EDF(tasks):
    # LCM of TT task periods
    T = lcm(*[obj.period for obj in tasks] )
    t=0
    sigma = ["idle"]*T
    R = [0]*len(tasks)
    WCRT = [0]*len(tasks)
    C = [obj.duration for obj in tasks]
    D = [obj.deadline for obj in tasks]
    #N = [obj.name for obj in tasks]gi
    D_copy = [obj.deadline for obj in tasks]
    while t < T:
        for i, task in enumerate(tasks):
            if C[i] > 0 and D[i] <= t:
                #print("t:", t, "deadline:", D[i], "C[i]:", C[i])
                return [], []
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
                #if (N[deadline_index] == 'polling_server'):
                #    self.poll_server.getTask()
                sigma[t] = deadline_index
                C[deadline_index] -= 1  
                #print("deadline index:", deadline_index, "C", C[deadline_index], D_copy)
                #print(" index of task:", earliest_deadline, D.index(earliest_deadline), D.index(D[deadline_index]), D)
                #print()
                if (C[deadline_index] == 0):
                    #print("reset", C[deadline_index])
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
        return [], []
    
    return sigma, WCRT

def EDP(ps: PollingServer):
    # alpha -> bandwidth
    # Delta -> delay
    # need to find out the way to calculate Cp, Tp, Dp
    Cp = ps.duration #=budget?
    Tp = ps.period
    Dp = ps.deadline
    delta = Tp + Dp -2 * Cp
    alpha = Cp / Tp
    
    #supply = 0
    #demand = 0
    
    P = [0]*len(ps.tasks)
    T = lcm(*[obj.period for obj in ps.tasks])
    Period = [obj.period for obj in ps.tasks]
    C = [obj.duration for obj in ps.tasks]
    D = [obj.deadline for obj in ps.tasks]
    #print(T)
    for i, task in enumerate(ps.tasks):
        t = 0
        responseTime = D[i] + 1
        while t <= T:
            supply = alpha * (t - delta)
            demand = 0
            for j, _ in enumerate(ps.tasks):
                if P[j] >= P[i]:
                    demand = demand + ceil(t / Period[j]) * C[j]
            
            if supply >= demand:
                responseTime = t
                break
            t += 1
        
        if responseTime > D[i]:
            return False, responseTime
    return True, responseTime
                    


if __name__ == "__main__":
    path = "./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv"
    dl = dataloader.DataLoader(path)
    TT, ET  = dl.loadFile()
    ps = PollingServer(duration=1800, period=2000, deadline=2000, tasks=ET)
    
    print(EDP(ps))
    # print(EDF(TT))
        