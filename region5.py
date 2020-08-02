# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:14:02 2020

@author: Rodrigo
"""

import sir
import matplotlib.pyplot as plt

#%%
imax = 0.1
umax = 0.3
gamma = 0.2
beta = 0.5

#%%
A = sir.SIR()
A.set_params([imax, umax, gamma, beta], flag = "bg")
P3 = A.add_point(0.9, 0.01)
P5 = A.add_point(0.9, 0.09)
A.find_regions()
A.get_trajectories()

#print(P3.trajectories)
#print(P5.trajectories)

fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax*1.1)
ax.plot(A.tau.s, A.tau.i)
ax.plot(A.phi.s, A.phi.i)
ax.plot(P3.trajectories[5].s, P3.trajectories[5].i)
ax.plot(P5.trajectories[5].s, P5.trajectories[5].i)

#B = sir.PlotSIR(A)
#B.show()
