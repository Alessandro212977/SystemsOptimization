import pandas as pd
from libraries.tasks import TT, ET


class DataLoader:
    def __init__(self, path) -> None:
        self.path = path

    def loadFile(self):
        data = pd.read_csv(self.path, sep=";")
        data.rename(columns = {'seperation':'separation'}, inplace = True)
        TTtasks = []
        ETtasks = []
        for __, row in data.iterrows():
            if row["type"] == "TT":
                TTtasks.append(TT(row["name"], row["duration"], row["period"], row["deadline"]))
            else:
                ETtasks.append(ET(row["name"], row["duration"], row["period"], row["deadline"], row["priority"], row["separation"]))

        return TTtasks, ETtasks


if __name__ == "__main__":
    test = DataLoader("./test_cases/taskset_small.csv")
    test.loadFile()
