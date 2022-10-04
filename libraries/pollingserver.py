class PollingServer:
    def __init__(self, budget, period, deadline, tasks) -> None:
        self.budget = budget
        self.period = period
        self.deadline = deadline
        self.tasks = tasks

    def getTask(self):
        pass