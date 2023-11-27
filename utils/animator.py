from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import time

class Animator:
    def __init__(self):
        self.fig, self.ax = plt.subplots()  
    
    def render(self, delay=0.05):
        clear_output(wait=True) # Clear output for dynamic display
        display(self.fig)       # Reset display
        time.sleep(delay)       

    def clear(self):
        self.ax.cla()

    def close(self):
        plt.close(self.fig)