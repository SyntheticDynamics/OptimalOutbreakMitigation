# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 09:43:01 2020

@author: Rodrigo
"""

import numpy as np
import sir
import matplotlib.pyplot as plt

imax = 0.1
umax = 0.5
gamma = 0.2
beta = 0.5

A = sir.SIR()
A.set_params(imax, umax, gamma, beta)
A.find_tau()
A.find_phi()
A.find_theta()
A.find_rho()

C = sir.CurveSegment(A.sbar, A.imax, 0, A, 0.7423377)
C.s = C.s + 0.05
#C = sir.CurveSegment(A.sbar+0.05, A.imax, 0, A, 0.74856325)
extra = np.linspace(C.s[-1], 1)
s0 = np.concatenate((C.s, extra))
i0_p1 = C.i
i0_p2 = np.array([0.005]*len(extra))
i0 = np.concatenate((i0_p1, i0_p2))

shortest_time_point = np.zeros(s0.shape)
shortest_time_traj = np.empty(s0.shape, dtype = object)

for ii in range(len(s0)):
    print(s0[ii], i0[ii])
    P = sir.Point(s0[ii], i0[ii], A)
    P.find_region()
    T = sir.TrajectoryCreator(P)
    T.get_trajectories()
    for t in T.trajectories:
        t.get_time()
    times = [t.time for t in T.trajectories]
    shortest = T.trajectories[times.index(min(times))]
    shortest_time_traj[ii] = shortest
    shortest_time_point[ii] = shortest.s[-1]


fig, ax = plt.subplots()
ax.set_xlim(0.4, 1)
ax.set_ylim(0, A.imax*1.1)
ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$I$")
ax.plot(A.tau.s, A.tau.i, "b-")
ax.plot(s0, i0, "g--")
ax.legend([r"$\tau$", "Initial Conditions"])
#fig.savefig("docs/reference_curve.pdf", format = "pdf")


fig, ax = plt.subplots()
ax.set_xlim(0.4, 1)
ax.set_ylim(0, A.imax*1.1)
ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$I$")
ax.plot(A.tau.s, A.tau.i, "b-")
ax.plot(shortest_time_traj[0].s, shortest_time_traj[0].i, "b--")
ax.plot(shortest_time_traj[9].s, shortest_time_traj[9].i, "r--")
ax.plot(shortest_time_traj[19].s, shortest_time_traj[19].i, "g--")
ax.plot(shortest_time_traj[29].s, shortest_time_traj[29].i, "y--")
ax.plot(shortest_time_traj[39].s, shortest_time_traj[39].i, "b:")
ax.plot(shortest_time_traj[49].s, shortest_time_traj[49].i, "r:")
ax.plot(shortest_time_traj[59].s, shortest_time_traj[59].i, "g:")
ax.plot(shortest_time_traj[69].s, shortest_time_traj[69].i, "y:")
ax.plot(shortest_time_traj[79].s, shortest_time_traj[79].i, "b-.")
ax.plot(shortest_time_traj[89].s, shortest_time_traj[89].i, "r-.")
ax.plot(shortest_time_traj[99].s, shortest_time_traj[99].i, "g-.")
#fig.savefig("docs/along_curve_path.pdf", format = "pdf")


fig, ax = plt.subplots()
ax.set_xlabel(r"$s_0$")
ax.set_ylabel(r"$s_f$")
ax.plot(s0[39:], shortest_time_point[39:], "b-")
ax.set_ylim(0.44, 0.7)
ax.plot(s0, s0, "g-")
ax.legend([r"$s_0 vs. s_f$", "Identidad"])
#fig.savefig("docs/along_curve_arrival.pdf", format = "pdf")


bending_point_s = np.zeros(shortest_time_traj.shape, dtype = float)
bending_point_i = np.zeros(shortest_time_traj.shape, dtype = float)
for ii in range(len(shortest_time_traj)):
    tra = shortest_time_traj[ii]
    max_i = max(tra.i)
    bending_point_i[ii] = max_i
    idx_i = np.where(tra.i == max(tra.i))[0][0]
    #print(idx_i)
    max_s = tra.s[idx_i]
    #print(max_s)
    bending_point_s[ii] = max_s

fig, ax = plt.subplots()
ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$i$")
ax.set_xlim(0.4, 1)
#ax.set_ylim(0.4, 1)
ax.set_ylim(0, A.imax*1.1)
ax.plot(A.tau.s, A.tau.i, "b-")
#ax.plot(s0, bending_point_s, "b-")
#ax.plot(s0, s0, "y-")
#ax.plot(s0, shortest_time_point, "r-")
ax.plot(bending_point_s, bending_point_i, "r-")
ax.legend([r"$\tau$", "Bending Point"])
fig.savefig("docs/bending_point.pdf", format = "pdf")
