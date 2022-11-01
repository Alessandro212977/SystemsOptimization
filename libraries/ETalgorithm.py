
from math import lcm, ceil
import dataloader
from pollingserver import PollingServer

class EDP:
    def __init__(self, path) -> None:
        dl = dataloader.DataLoader(path)
        self.TT, self.ET  = dl.loadFile()
        self.PS = self.initialize() #list of polling servers

    def initialize(self):
        return [PollingServer(5000, 2000, 2000, self.ET)]

    def run(self):
        # alpha -> bandwidth
        # Delta -> delay
        # need to find out the way to calculate Cp, Tp, Dp
        Cp = self.PS[0].budget
        Tp = self.PS[0].period
        Dp = self.PS[0].deadline
        delta = Tp + Dp -2 * Cp
        alpha = Cp / Tp
        
        #supply = 0
        #demand = 0
        
        P = [0]*len(self.ET)
        T = lcm(*[obj.period for obj in self.ET])
        Period = [obj.period for obj in self.ET]
        C = [obj.duration for obj in self.ET]
        D = [obj.deadline for obj in self.ET]
        #print(T)
        for i, task in enumerate(self.ET):
            t = 0
            responseTime = D[i] + 1
            while t <= T:
                supply = alpha * (t - delta)
                demand = 0
                for j, _ in enumerate(self.ET):
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
    edp = EDP("./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv")
    print(edp.run())
        