# importing the required module
import matplotlib.pyplot as plt
  
# # x axis values
x = [1, 1, 1, 1, 6, 6, 6, 8, 8, 9, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 22, 22, 22, 22, 22, 22, 27, 27, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 16, 16, 16, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 26, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 'Idle', 'Idle', 'Idle', 'Idle', 'Idle', 'Idle', 'Idle']
ts = 0

    
# # corresponding y axis values
# y = [T1, T2, T3, T4, T5, T6]
  
# # plotting the points 
# plt.plot(x, y)
  
# # naming the x axis
# plt.xlabel('x - axis')
# # naming the y axis
# plt.ylabel('y - axis')
  
# # giving a title to my graph
# plt.title('My first graph!')
  
# # function to show the plot
# plt.show()


    
    


import plotly.figure_factory as ff
from EDFalgorithm import EDF as edf
df = []
for i in x:
    duration = edf.getDuration("tTT"+str(i))
    df.append(dict(Task = "Task"+str(i), start = 0, Finish = duration -1), Resource = 'Complete')
    


# df = [dict(Task="Job-1", Start='2017-01-01', Finish='2017-02-02', Resource='Complete'),
#       dict(Task="Job-1", Start='2017-02-15', Finish='2017-03-15', Resource='Incomplete'),
#       dict(Task="Job-2", Start='2017-01-17', Finish='2017-02-17', Resource='Not Started'),
#       dict(Task="Job-2", Start='2017-01-17', Finish='2017-02-17', Resource='Complete'),
#       dict(Task="Job-3", Start='2017-03-10', Finish='2017-03-20', Resource='Not Started'),
#       dict(Task="Job-3", Start='2017-04-01', Finish='2017-04-20', Resource='Not Started'),
#       dict(Task="Job-3", Start='2017-05-18', Finish='2017-06-18', Resource='Not Started'),
#       dict(Task="Job-4", Start='2017-01-14', Finish='2017-03-14', Resource='Complete')]

colors = {'Not Started': 'rgb(220, 0, 0)',
          'Incomplete': (1, 0.9, 0.16),
          'Complete': 'rgb(0, 255, 100)'}

fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,
                      group_tasks=True)
fig.show()