import pandas as pd
from pandas import read_csv
from libraries.tasks import TT, ET
from os import listdir
from os.path import join
import sys
import os


def parseTests(self):
        path = os.path.abspath(os.path.dirname(__file__))
        data = pd.read_csv(path, sep=";")
        data.rename(columns = {'seperation':'separation'}, inplace = True)
        TTtasks = []
        ETtasks = []
        for __, row in data.iterrows():
            if row["type"] == "TT":
                TTtasks.append(TT(row["name"], row["duration"], row["period"], row["deadline"]))              
            else:
                ETtasks.append(ET(row["name"], row["duration"], row["period"], row["deadline"], row["priority"], row["separation"]))

        return TTtasks, ETtasks

def loadTests(start: int, end: int):
    testFolder = "test_cases"
    i: int = -1
    for folder in listdir(testFolder):
        subFolder = join(testFolder, folder)
        for filename in listdir(testFolder):
            i += 1
            if i < start or i >= end: 
                continue
            
            # yield (*parseTests(read_csv(filename, sep=";")), filename)
            yield (*parseTests(read_csv(join(subFolder, filename), sep=";")), subFolder, filename)