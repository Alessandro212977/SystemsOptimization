# load task from file


from math import gcd 


def load_task():
    lines = []
    tasks = []
    #rawdata = open('asset/tasks.txt', 'r')
    #rawdata.seek(45)
    tt_data = open('asset/tt.txt','r')
    tt_data.seek(10)
    data = tt_data.readlines()
    for d in data:
        lines.append(d.strip('\n'))
    for line in lines:
        tasks.append(list(line.split(",")))
     
    return tasks
    
def getLcm(tasks):
    temp  = []
    
    for i in range(len(tasks)):
        temp.append(int(tasks[i][2]))
    lcm = temp[0]
    for t in temp[1:]:
        lcm = int(lcm * t/gcd(lcm, t))

    return lcm
    
    

def edf():
    pass







tasks = load_task()
print(tasks)
lcm = getLcm(tasks)
print(lcm)