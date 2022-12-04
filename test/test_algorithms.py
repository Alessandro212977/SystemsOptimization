import unittest
from math import ceil, floor, lcm
from libraries.algorithms import EDF, EDP, extention1
from libraries.tasks import PollingServer
import libraries.dataloader as dataloader


class TestEDF(unittest.TestCase):
    def load(self, name, path="./test/test_taskset/"):
        dl = dataloader.DataLoader(path + name)
        return dl.loadFile()

    def test_EDF1(self):
        TT, ET = self.load("EDF_test_1.csv")
        schedulable, __, __, __ = EDF(TT)
        self.assertFalse(schedulable)

    def test_EDF2(self):
        TT, ET = self.load("EDF_test_2.csv")
        schedulable, __, __, __ = EDF(TT)
        self.assertTrue(schedulable)

    def test_EDF3(self):
        TT, ET = self.load("EDF_test_3.csv")
        schedulable, __, __, __ = EDF(TT)
        self.assertFalse(schedulable)

    def test_EDF4(self):
        TT, ET = self.load("EDF_test_4.csv")
        __, __, wcrt, __ = EDF(TT)
        self.assertListEqual(wcrt, [20, 40])

    def test_EDF5(self):
        TT, ET = self.load("EDF_test_5.csv")
        __, __, wcrt, __ = EDF(TT)
        self.assertListEqual(wcrt, [35, 4, 8, 48, 30])
    
    def test_EDF6(self):
        TT, ET = self.load("EDF_test_6.csv")
        __, __, wcrt, penalty = EDF(TT)
        self.assertListEqual(wcrt+[penalty], [50, 80, 1/8])


class TestEDP(unittest.TestCase):
    def load(self, name, path="./test/test_taskset/"):
        dl = dataloader.DataLoader(path + name)
        return dl.loadFile()

    def test_EDP1(self):
        TT, ET = self.load("EDP_test_1.csv")
        ps = PollingServer(name="ps1", duration=100, period=1000, deadline=1000, tasks=ET, separation=1)
        schedulable, _, _ = EDP(ps)
        # print("EDP\n", EDP(ps))
        self.assertTrue(schedulable)

    def test_EDP2(self):
        TT, ET = self.load("EDP_test_2.csv")
        ps = PollingServer(name="ps1", duration=100, period=1000, deadline=1000, tasks=ET, separation=1)
        schedulable, _, _ = EDP(ps)
        self.assertTrue(schedulable)

"""
class TestExtension1(unittest.TestCase):
    def load(self, name, path="./test/test_taskset/"):
        dl = dataloader.DataLoader(path + name)
        return dl.loadFile()

    def test_ex1_1(self):
        TT, ET = self.load("Extension1_1.csv")
        # print(extention1(TT))
        self.assertTrue(extention1(TT))

    def test_ex1_2(self):
        TT, ET = self.load("Extension1_2.csv")
        # print(extention1(TT))
        self.assertTrue(extention1(TT))

    def test_ex1_3(self):
        TT, ET = self.load("Extension1_3.csv")
        self.assertTrue(extention1(TT))    

    def test_ex1_4(self):
        TT, ET = self.load("Extension1_4.csv")
        ps = PollingServer(name="ps1", duration=1000, period=5000, deadline=2000, tasks=ET, separation=1)
        self.assertTrue(extention1(TT + [ps]))

    def test_ex1_5(self):
        TT, ET = self.load("Extension1_5.csv")
        ps = PollingServer(name="ps1", duration=100, period=1000, deadline=1000, tasks=ET, separation=1)
        self.assertTrue(extention1(TT + [ps]))
"""

if __name__ == "__main__":
    unittest.main()
