# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:29:42 2020

@author: Rodrigo
"""

import sir
import matplotlib.pyplot as plt

imax = 0.1
umax = 0.5
gamma = 0.2
beta = 0.5

A = sir.SIR()
A.set_params(imax, umax, gamma, beta)
A.find_curves()
P2 = A.add_point(0.6, 0.08)
P3 = A.add_point(0.9, 0.09)
A.find_regions()

print(P2.region)
print(P3.region)

A.get_trajectories()
P2.get_times()
P3.get_times()
P2.get_i_times()
P3.get_i_times()

fig, ax = plt.subplots()
ax.plot(P2.times / P2.i_times)
ax.plot(P3.times / P3.i_times)
