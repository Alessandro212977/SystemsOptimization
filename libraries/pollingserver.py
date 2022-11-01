class PollingServer:
    def __init__(self, duration, period, deadline, tasks, name) -> None:
        self.duration = duration
        self.period = period
        self.deadline = deadline
        self.tasks = tasks
        self.name = name

    def getTask(self):
        for obj in self.tasks:
            if obj.period == self.period:
                return obj