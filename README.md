# Systems Optimization course
## Configuring ADAS Applications with Time-Triggered and Event-Triggered Tasks
Keywords and phrases time-triggered, event-triggered, scheduling, real-time, combinatorial opti-
mization

# Goal
As an input you are given: 
 - (1) An application model consisting of a set of TT tasks and a set of ET tasks,
 - (2) An architecture model consisting of one core that schedules TT tasks with timeline scheduling and ET tasks with polling servers
 
You will have to design and implement an optimization algorithm that determines an optimized solution which consists of the following: 
 - (1) The number of polling servers, which then become extra TT tasks. 
 - (2) For each polling server (task), the period, budget, and deadline.
 - (3) Which sub-sets of ET tasks are handled within the respective polling servers.
 - (4) A TT schedule such that also the TT tasks are schedulable.
 
The solution should be optimized such that: 
 - (1) Both the TT and ET tasks are schedulable, i.e., they complete before their deadlines.
 - (2) The ET task separation constraints are satisfied.
 - (3) The average worst-case response times (WCRT) of all tasks (TT and ET) is minimized.
 
 # Solution
 TODO
 
 # How to...
 TODO
