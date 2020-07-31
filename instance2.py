# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:32:30 2020

@author: Rodrigo
"""

import numpy as np
import matplotlib.pyplot as plt
import sir

imax = 0.1
gamma = 0.2
beta = 0.5
umax = 0.5

s2 = 0.6
i2 = 0.08

s3 = 1 - 1e-2
i3 = 1e-2

s4 = 0.8
i4 = 1e-4

s5 = 0.95
i5 = 0.095

#%%
A = sir.SIR()
A.set_params(imax, umax, gamma, beta)
A.find_curves()

P2 = A.add_point(s2, i2)
P3 = A.add_point(s3, i3)
P4 = A.add_point(s4, i4)
A.find_regions()
A.get_trajectories()
P2.get_times()
P3.get_times()
P4.get_times()
P2s = P2.get_least_time()
P3s = P3.get_least_time()
P4s = P4.get_least_time()

fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, A.imax*1.1)
ax.plot(A.tau.s, A.tau.i)
ax.plot(P2s.s, P2s.i)
ax.plot(P3s.s, P3s.i)
ax.plot(P4s.s, P4s.i)

#B = sir.PlotSIR(A)
#B.show()
"""
C2 = sir.Point(s2, i2, A)
C2.find_region()

C3 = sir.Point(s3, i3, A)
C3.find_region()

C4 = sir.Point(s4, i4, A)
C4.find_region()

C5 = sir.Point(s5, i5, A)
C5.find_region()

T2 = sir.TrajectoryCreator(C2)
t2 = T2.get_trajectories()

T3 = sir.TrajectoryCreator(C3)
t3 = T3.get_trajectories()

T4 = sir.TrajectoryCreator(C4)
t4 = T4.get_trajectories()

T5 = sir.TrajectoryCreator(C5)
t5 = T5.get_trajectories()


#%%
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, A.imax * 1.1)
ax.set_xlabel("S")
ax.set_ylabel("I")
ax.plot(A.tau.s, A.tau.i, "b-")
ax.plot(A.phi.s, A.phi.i, "r-")
ax.plot(A.theta.s, A.theta.i, "y-")
ax.plot(A.rho.s, A.rho.i, "k-")
ax.plot([0, 1], [A.imax]*2, "k--")
#ax.plot(T3.main_trajectory.segments[0].s,
#        T3.main_trajectory.segments[0].i, "g--")
#ax.plot(T3.main_trajectory.segments[1].s,
#        T3.main_trajectory.segments[1].i, "b--")
#ax.plot(T3.main_trajectory.segments[2].s,
#        T3.main_trajectory.segments[2].i, "r--")
#ax.plot(T3.u0_phi_intersection[0], T3.u0_phi_intersection[1], "b.")
ax.plot(C2.s0, C2.i0, "rx")
ax.plot(t2[0].s, t2[0].i, "r--")
ax.plot(t2[30].s, t2[30].i, "b--")
ax.plot(t2[60].s, t2[60].i, "g--")
ax.plot(t2[80].s, t2[80].i, "y--")
fig.savefig("docs/trajectories.pdf", format = "pdf")
#ax.plot(down_s, down_i, "rx")

#%%
for trajectory in t2:
    trajectory.get_time()

times2 = [trajectory.time for trajectory in t2]
arrival2 = [trajectory.s[-1] for trajectory in t2]

for trajectory in t3:
    trajectory.get_time()

times3 = [trajectory.time for trajectory in t3]
arrival3 = [trajectory.s[-1] for trajectory in t3]

for trajectory in t4:
    trajectory.get_time()

times4 = [trajectory.time for trajectory in t4]
arrival4 = [trajectory.s[-1] for trajectory in t4]

mint2 = min(times2)
mint3 = min(times3)
mint4 = min(times4)
shortest2 = t2[times2.index(mint2)]
shortest3 = t3[times3.index(mint3)]
shortest4 = t4[times4.index(mint4)]

fig, ax = plt.subplots()
#ax.plot(arrival2, times2)
#ax.plot(arrival3, times3)
ax.plot(arrival4, times4)
ax.set_xlabel(r"Coordenada de llegada en $s$.")
ax.set_ylabel(r"$t$")
ax.set_yscale("log")
ax.set_title("Tiempos de llegada de punto con origen en 4.")
#fig.savefig("docs/tiempos_2.pdf", format = "pdf")

#%%
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, A.imax * 1.1)
ax.set_xlabel("S")
ax.set_ylabel("I")
ax.plot(A.tau.s, A.tau.i, "b-")
ax.plot(A.phi.s, A.phi.i, "r-")
ax.plot(A.theta.s, A.theta.i, "y-")
ax.plot(A.rho.s, A.rho.i, "k-")
ax.legend([r"$\tau$", r"$\phi$", r"$\theta$", r"$\rho$"])
ax.plot([0, 1], [A.imax, A.imax], "r--")
for i in range(1000):
    s = np.random.rand(1)
    i = np.random.rand(1)*A.imax*1.1
    Px = sir.Point(s, i, A)
    Px.find_region()
    ax.plot(s, i, "{}.".format(["b", "r", "g", "y", "k"][Px.region-1]))
#fig.savefig("docs/regions.pdf", format = "pdf")
"""