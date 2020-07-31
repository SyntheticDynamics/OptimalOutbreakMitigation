# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:39:52 2020

@author: Rodrigo
"""

import numpy as np
import scipy.integrate

def dx(x, t):
    return -0.5*x

def dt(x, t):
    return 1

def F(x, t):
    return np.array([dx(x, t), dt(x, t)])

x0 = np.array([1, 0])
y = F(x0[0], 0)
print(y)
t_range = np.linspace(0, 2)
series = scipy.integrate.odeint(F, y0 = x0, t = t_range, tfirst = False)
