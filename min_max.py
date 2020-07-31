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
A.set_params(imax, umax, gamma, beta)
A.find_tau()
A.find_phi()
A.find_theta()
A.find_rho()

start = A.tau._curve_sol(i_ref = i0)
s0 = np.linspace(start[0], 1, 100)

#B = sir.PlotSIR(A)
#B.show()

arr_point = []
arr_tra = []

for s in s0:
    P = sir.Point(s, i0, A)
    P.find_region()
    if P.region == 1 or P.region == 5:
        continue
    T = sir.TrajectoryCreator(P)
    trajectories = T.get_trajectories()
    #print(trajectories)
    for tra in trajectories:
        tra.get_time()
    times = [tra.time for tra in trajectories]
    arrival = [tra.s[-1] for tra in trajectories]
    min_arr = arrival[times.index(min(times))]
    min_arr_tra = trajectories[times.index(min(times))]
    arr_point.append(min_arr)
    arr_tra.append(min_arr_tra)

fig, ax = plt.subplots()
ax.plot(s0[1:], arr_point)
ax.plot(s0, s0)
ax.set_xlabel(r"$s_0$")
ax.set_ylabel(r"$s_f$")
#ax.set_xlim(0, 1)
ax.set_ylim(0.57, 0.68)
ax.legend([r"$s_0 vs. s_f$", "Identidad"])
fig.savefig("docs/s0_vs_sf.pdf", format = "pdf")

fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, A.imax * 1.1)
ax.set_xlabel("S")
ax.set_ylabel("I")
ax.plot(A.tau.s, A.tau.i, "b-")
ax.plot(A.phi.s, A.phi.i, "r-")
ax.plot(A.theta.s, A.theta.i, "y-")
ax.plot(A.rho.s, A.rho.i, "k-")
ax.plot(arr_tra[1].s, arr_tra[1].i, "r--")
ax.plot(arr_tra[5].s, arr_tra[5].i, "b--")
ax.plot(arr_tra[10].s, arr_tra[10].i, "g--")
ax.plot(arr_tra[15].s, arr_tra[15].i, "y--")
ax.plot(arr_tra[20].s, arr_tra[20].i, "r:")
ax.plot(arr_tra[28].s, arr_tra[28].i, "b:")
ax.plot(arr_tra[35].s, arr_tra[35].i, "g:")
ax.plot(arr_tra[41].s, arr_tra[41].i, "y:")
ax.plot(arr_tra[47].s, arr_tra[47].i, "r-.")
ax.plot(arr_tra[61].s, arr_tra[61].i, "b-.")
ax.plot(arr_tra[78].s, arr_tra[78].i, "g-.")
ax.plot(arr_tra[98].s, arr_tra[98].i, "y-.")
fig.savefig("docs/short_trajectories.pdf", format = "pdf")
