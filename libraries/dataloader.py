import pandas as pd
from tasks import TT, ET

class DataLoader:

    def __init__(self, path) -> None:
        self.path = path

    def loadFile(self):
        data = pd.read_csv(self.path, sep = ";")
        TTtasks = []
        ETtasks = []
        for __, row in data.iterrows():
            if row["type"] == "TT":
                TTtasks.append(TT(row["name"], row["duration"], row["period"], row["deadline"]))
            else:
                ETtasks.append(ET(row["name"], row["duration"], row["period"], row["deadline"], row["priority"]))
        return TTtasks, ETtasks


if __name__ == "__main__":
    test = DataLoader("./test_cases/inf_10_10/taskset__1643188013-a_0.1-b_0.1-n_30-m_20-d_unif-p_2000-q_4000-g_1000-t_5__0__tsk.csv")
    test.loadFile()