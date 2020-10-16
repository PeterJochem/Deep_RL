import numpy as np


"""Class represents the trajectory generator"""
class TG:

    def __init__(self, a_x, a_y):
        
        self.a_x = a_x
        self.a_y = a_y
        self.t = 0
        

    """TG's control signal"""
    def U_TG(self):
        
        u_x = self.a_x * np.sin(2 * np.pi * self.t)
        u_y = (self.a_y/2.0) * (np.sin(2 * np.pi * self.t) * np.cos(2 * np.pi * self.t))
     
        return u_x, u_y

