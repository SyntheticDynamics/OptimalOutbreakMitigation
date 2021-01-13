#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:51:04 2020

@author: rmm
"""

import sir
import matplotlib.pyplot as plt

imax = 0.1
umax = 0.6
r = 1.73
Dyn = sir.SIR()
Dyn.set_params([imax, umax, r], flag = "r")
sir.find_commutation_curve(Dyn)

"""
end = Dyn.phi._curve_sol(5e-4)[0]
s0, i0 = sir.create_initial_conditions(Dyn, 1e-3, 5e-4, end = end)

for s, i in zip(s0, i0):
    Dyn.add_point(s, i)

Dyn.find_regions()
regs = [p.region for p in Dyn.points]

for p in Dyn.points:
    Mx = sir.MinimumTrajectory(p, Dyn)
    Mx.find_commutation()
    p.least_time = Mx.trajectory
"""

#%%
fig, ax = plt.subplots(figsize = (18, 12))
ax.set_xlim(Dyn.sbar, 2)
ax.set_ylim(0, Dyn.imax * 1.1)
ax.plot(Dyn.tau.s, Dyn.tau.i, "b-", alpha = 0.5)
ax.plot(Dyn.phi.s, Dyn.phi.i, "r-", alpha = 0.5)
ax.plot(Dyn.theta.s, Dyn.theta.i, "g-", alpha = 0.5)
ax.plot(Dyn.commutation_curve[0], Dyn.commutation_curve[1], "k-", alpha = 0.5)
#ax.plot(s0, i0, "r-")
