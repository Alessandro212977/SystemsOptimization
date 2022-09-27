class Event:
    def __init__(self, name, duration, period, deadline) -> None:
        self.name = name
        self.duration = duration
        self.period = period
        self.deadline = deadline

class TT(Event):
    def __init__(self, name, duration, period, deadline) -> None:
        super().__init__(name, duration, period, deadline)
    

class ET(Event):
    def __init__(self, name, duration, period, deadline, priority) -> None:
        super().__init__(name, duration, period, deadline)
        self.priority = priority