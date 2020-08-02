# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:31:50 2020

@author: Rodrigo
"""

import numpy as np
import sir
import matplotlib.pyplot as plt

imax = 0.1
umax = 0.5
gamma = 0.2
beta = 0.5

i0 = 0.04

A = sir.SIR()
A.set_params([imax, umax, gamma, beta], flag = "bg")

start = A.tau._curve_sol(i_ref = i0)
s0 = np.linspace(start[0], 1, 100)

#B = sir.PlotSIR(A)
#B.show()

for s in s0:
    A.add_point(s, i0)

min_tra = np.empty([len(A.points), ], dtype = object)
A.find_regions()
A.get_trajectories()
for ii in range(len(A.points)):
    p = A.points[ii]
    if p.region == 5:
        continue
    p.get_times()
    min_tra[ii] = p.get_least_time()

arr_point = [trajectory.s[-1] for trajectory in min_tra]

fig, ax = plt.subplots()
ax.plot(s0, arr_point)
ax.plot(s0, s0)
ax.set_xlabel(r"$s_0$")
ax.set_ylabel(r"$s_f$")
#ax.set_xlim(0, 1)
ax.set_ylim(0.57, 0.68)
ax.legend([r"$s_0 vs. s_f$", "Identidad"])
#fig.savefig("docs/s0_vs_sf.pdf", format = "pdf")

fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, A.imax * 1.1)
ax.set_xlabel("S")
ax.set_ylabel("I")
ax.plot(A.tau.s, A.tau.i, "b-")
ax.plot(A.phi.s, A.phi.i, "r-")
ax.plot(A.theta.s, A.theta.i, "y-")
ax.plot(A.rho.s, A.rho.i, "k-")
ax.plot(min_tra[1].s, min_tra[1].i, "r--")
ax.plot(min_tra[5].s, min_tra[5].i, "b--")
ax.plot(min_tra[10].s, min_tra[10].i, "g--")
ax.plot(min_tra[15].s, min_tra[15].i, "y--")
ax.plot(min_tra[20].s, min_tra[20].i, "r:")
ax.plot(min_tra[28].s, min_tra[28].i, "b:")
ax.plot(min_tra[35].s, min_tra[35].i, "g:")
ax.plot(min_tra[41].s, min_tra[41].i, "y:")
ax.plot(min_tra[47].s, min_tra[47].i, "r-.")
ax.plot(min_tra[61].s, min_tra[61].i, "b-.")
ax.plot(min_tra[78].s, min_tra[78].i, "g-.")
ax.plot(min_tra[98].s, min_tra[98].i, "y-.")
#fig.savefig("docs/short_trajectories.pdf", format = "pdf")
