class PollingServer:
    def __init__(self, duration, period, deadline, tasks) -> None:
        self.duration = duration
        self.period = period
        self.deadline = deadline
        self.tasks = tasks

    def getTask(self):
        for obj in self.tasks:
            if obj.period == self.period:
                return obj