
class Event:
    def __init__(self, name, duration, period, deadline) -> None:
        self.name = name
        self.duration = duration
        self.period = period
        self.deadline = deadline

    def __repr__(self):
       return "{}: duration: {}, period: {}, deadline: {}".format(self.name, self.duration, self.period, self.deadline)
       
class TT(Event):
    def __init__(self, name, duration, period, deadline) -> None:
        super().__init__(name, duration, period, deadline)
    

class ET(Event):
    def __init__(self, name, duration, period, deadline, priority) -> None:
        super().__init__(name, duration, period, deadline)
        self.priority = priority

class PollingServer(Event):
    def __init__(self, name, duration, period, deadline, tasks) -> None:
        super().__init__(name, duration, period, deadline)
        self.tasks = tasks
