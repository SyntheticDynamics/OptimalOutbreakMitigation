# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:39:31 2020

@author: Rodrigo
"""

import numpy as np
import scipy.integrate
import sir

#%%
class Test():
    def __init__(self, x):
        self.x = x
        self.a = 3
        self.x_start = 0
        self.x_end = 2
    
    
    def fun(self, x = None, gamma = 0.2,
            beta = 0.5, u = 0.5, s0 = 0.6, i0 = 0.08):
        if x is None:
            x = self.x
        f = ((gamma/((1 - u)*beta))*np.log(x/s0)) - (x - s0) + i0
        return f
    
    def fun_int(self, x = None):
        if x is None:
            x = self.x
        return 1 / (x * self.fun(x))
    
    def get_time(self, start = None, end = None):
        if start is None:
            start = self.x_start
        if end is None:
            end = self.x_end
        area = scipy.integrate.quad(self.fun_int, start, end)
        area = area[0] / ((1 - 0.5)*0.5)
        return -area


def fun(x):
    f = ((0.2/((1 - 0.5)*0.5))*np.log(x/0.6)) - (x - 0.6) + 0.08
    return f

def fun2(x):
    f = 1 / (x * fun(x))
    return f

#%%
A = Test(0.6)
print(A)
print(A.x)
print(A.get_time(0.6, 0.5))


imax = 0.1
umax = 0.5
gamma = 0.2
beta = 0.5
B = sir.SIR()
B.set_params(imax, umax, gamma, beta)
B.find_tau()
T = B.tau
T.get_time(2, 1)

L = sir.CurveSegment(0.6, 0.08, umax, B, 0.5)
L.get_time(0.6, 0.5)
print(L.time)

i = scipy.integrate.quad(fun2, 0.6, 0.5)
print(-i[0] / ((1 - 0.5)*0.5))
