from tkinter import *
from libraries.graphplot import *
class MyWindow:
    def __init__(self, win):
    
        
        self.lbl1=Label(win, text='Choose one Algorithm to Plot', font=('Times', 24), background="black",
                 foreground="red")
        self.v0=IntVar()
        self.v0.set(1)
        self.r1=Radiobutton(window, text="Simulated Anealing", variable=self.v0,value=1)
        self.r2=Radiobutton(window, text="TT Task", variable=self.v0,value=2)
        self.b1=Button(win, text='Plot',  fg = 'blue', command=self.add, height = 5, width = 30)
        
        self.lbl1.place(x=60, y=30)
        self.b1.place(x=50, y=200)
        self.r1.place(x=60,y=120)
        self.r2.place(x=240, y=120)       

    def add(self):
        if self.v0.get() == 1:
            run("SA")
        else:
            run("TT")
            
window=Tk()
mywin=MyWindow(window)
window.title('System Optimization')
window_width = 400
window_height = 300

# get the screen dimension
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# find the center point
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)

# set the position of the window to the center of the screen
window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
#window.geometry("400x300+10+10")

window.mainloop()