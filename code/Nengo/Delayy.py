import numpy as np

class Del:
   
    def __init__(self,timedelay):
        self.history = np.zeros(timedelay)

    def step(self,x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        return self.history[0]