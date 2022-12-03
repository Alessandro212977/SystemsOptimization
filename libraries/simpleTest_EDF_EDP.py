from libraries.algorithms import EDF, EDP
from libraries.tasks import PollingServer
import libraries.dataloader as dataloader
from pathlib import Path
from math import lcm

def test_EDF1():
    path = "./unit_testing_tasks/EDF_test_1.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()
    schedulable, __, __, __= EDF(TT)
    return schedulable == False

def test_EDF2():
    path = "./unit_testing_tasks/EDF_test_2.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()
    schedulable, __, __, __= EDF(TT)
    return schedulable == True

def test_EDF3():
    path = "./unit_testing_tasks/EDF_test_3.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()
    schedulable, __, __, __= EDF(TT)
    return schedulable == False

def test_EDF4():
    path = "./unit_testing_tasks/EDF_test_4.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()
    schedulable, __, wcrt, __= EDF(TT)
    # print(wcrt)
    expected_wcrt = [20,40]
    for i,j in zip(expected_wcrt, wcrt):
        if i!=j:
            return False       
    return True

def test_EDF5():
    path = "./unit_testing_tasks/EDF_test_5.csv"
    dl = dataloader.DataLoader(path)
    TT, ET = dl.loadFile()
    schedulable, __, wcrt, __= EDF(TT)
    # print(wcrt, schedulable)as
    expected_wcrt = [35, 4, 8, 48, 30]
    for i,j in zip(expected_wcrt, wcrt):
        if i!=j:
            return False       
    return True


def test_EDP1():
    path = "./unit_testing_tasks/EDP_test_1.csv"
    dl = dataloader.DataLoader(path)
    TT, ET= dl.loadFile()
    ps = PollingServer(name='ps1', duration=100, period=1000, deadline=1000, tasks=ET, separation=1)
    schedulable, _, _ = EDP(ps)
    # print("EDP\n", EDP(ps))
    return schedulable == True

def test_EDP2():
    path = "./unit_testing_tasks/EDP_test_2.csv"
    dl = dataloader.DataLoader(path)
    TT, ET= dl.loadFile()
    ps = PollingServer(name='ps1', duration=100, period=1000, deadline=1000, tasks=ET, separation=1) 
    schedulable, _, _ = EDP(ps)
    return schedulable == True


if __name__ == "__main__":
    print(f"test EDF 1 passed: {test_EDF1()}")
    print(f"test EDF 2 passed: {test_EDF2()}")
    print(f"test EDF 3 passed: {test_EDF3()}")
    print(f"test EDF 4 passed: {test_EDF4()}")
    print(f"test EDF 5 passed: {test_EDF5()}")
    print(f"test EDP 1 passed: {test_EDP1()}")
    print(f"test EDP 2 passed: {test_EDP2()}")


    """
    path1 = "./test_cases/EDF_test_0.csv"
    dl = dataloader.DataLoader(path1)
    TT1, ET1= dl.loadFile()

    path2 = "./test_cases/EDP_test_0.csv"
    dl = dataloader.DataLoader(path2)
    TT2, ET2= dl.loadFile()

    schedulable, timetable, wcrt, penalty = EDF(TT1)
    # hyperperiod = lcm(*[tt.period for tt in TT])
    # print("hyperperiod: ", hyperperiod)
    print("EDF\n", schedulable, wcrt, penalty)

    ps = PollingServer(name='ps1', duration=3000, period=3000, deadline=3000, tasks=ET2, separation=1)
    
    print("EDP\n", EDP(ps))
    """


