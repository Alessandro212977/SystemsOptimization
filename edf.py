# load task from file
def load_task():
    temp = []
    #rawdata = open('asset/tasks.txt', 'r')
    #rawdata.seek(45)
    tt_data = open('asset/tt.txt','r')
    tt_data.seek(10)
    lines = tt_data.readlines()
    
    for line in lines:
        temp.append(line.strip('\n'))
    
    
    
    print(temp)
def getLcm():
    pass
    

def edf():
    pass
    

load_task()