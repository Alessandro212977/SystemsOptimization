import unittest

from libraries.algorithms import EDF, EDP
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


if __name__ == "__main__":
    unittest.main()
