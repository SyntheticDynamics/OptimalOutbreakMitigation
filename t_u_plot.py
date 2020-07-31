# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 09:36:10 2020

@author: Rodrigo
"""

import sir
import matplotlib.pyplot as plt

#%%
imax = 0.1
umax = 0.5
gamma = 0.2
beta = 0.5

#%%
A = sir.SIR()
A.set_params([imax, umax, gamma, beta], flag = "bg")
A._find_curves()
P = A.add_point(0.9, 0.00001)
A.find_regions()

A.get_shortest()

T = P.least_time
print(T.segments)
T.plot_time()
