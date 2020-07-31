# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:52:01 2020

@author: Rodrigo
"""

import sir
import numpy as np
import matplotlib.pyplot as plt

#%%
def find_max(trajectory):
    """
    """
    x = trajectory.s
    y = trajectory.i
    yt = np.abs(y - max(y))
    #print(yt, type(yt))
    yt = yt < 1e-5
    #print(yt, type(yt))
    max_idx = np.where(yt == True)[0]
    #print(max_idx)
    max_idx = max(max_idx)
    return [x[max_idx], y[max_idx]]

#%%
imax = 0.1
umax = 0.5
gamma = 0.2
beta = 0.5

displacement = 0.01
i_low = 0.0005

#%%
A = sir.SIR()
A.set_params(imax, umax, gamma, beta)
A.find_curves()

C = sir.CurveSegment(A.sbar, A.imax, 0, A, 1)
C.s = C.s + displacement
s_inter, i_inter = C._curve_sol(i_low)
C = sir.CurveSegment(A.sbar, A.imax, 0, A, s_inter)
C.s = C.s + displacement
s0 = np.linspace(C.s[-1], 1)
i0 = np.array([i_inter]*len(s0))
s0 = np.concatenate((C.s, s0))
i0 = np.concatenate((C.i, i0))

fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax*1.1)
ax.plot(A.tau.s, A.tau.i, "r-")
ax.plot(s0, i0, "b-")

#%%
for ii in range(len(s0)):
    A.add_point(s0[ii], i0[ii])
A.find_regions()
A.get_trajectories()
A.get_shortest()

least_time = np.empty([len(A.points), ], dtype = object)
least_intervention = np.empty(np.shape(least_time), dtype = object)
for ii in range(len(A.points)):
    p = A.points[ii]
    p.get_times()
    p.get_i_times()
    least_time[ii] = p.get_least_time()
    least_intervention[ii] = p.get_least_intervention()

final_point_t = [tra.s[-1] for tra in least_time]
final_point_i = [tra.s[-1] for tra in least_intervention]

fig, ax = plt.subplots()
ax.set_xlabel(r"$s_0$")
ax.set_ylabel(r"$s_f$")
ax.plot(s0, final_point_t, "r-")
ax.plot(s0, final_point_i, "b-")
ax.legend(["Least Time", "Least Intervention Time"])
#fig.savefig("docs/lt_vs_li.pdf", format = "pdf")
#fig.savefig("docs/lt_vs_li.jpg", format = "jpg")

fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax*1.1)
ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$I$")
ax.plot(A.tau.s, A.tau.i)
ax.plot(A.phi.s, A.phi.i)
ax.plot(least_intervention[0].s, least_intervention[0].i)
ax.plot(least_intervention[9].s, least_intervention[9].i)
ax.plot(least_intervention[19].s, least_intervention[19].i)
ax.plot(least_intervention[29].s, least_intervention[29].i)
ax.plot(least_intervention[39].s, least_intervention[39].i)
ax.plot(least_intervention[49].s, least_intervention[49].i)
ax.plot(least_intervention[59].s, least_intervention[59].i)
ax.plot(least_intervention[69].s, least_intervention[69].i)
ax.plot(least_intervention[79].s, least_intervention[79].i)
ax.plot(least_intervention[89].s, least_intervention[89].i)
ax.plot(least_intervention[99].s, least_intervention[99].i)
ax.legend([r"$\tau$", r"$\phi$"])
#fig.savefig("docs/min_int.pdf", format = "pdf")
#fig.savefig("docs/min_int.jpg", format = "jpg")

#%%
bp_s = np.zeros([len(least_intervention), ])
bp_i = np.zeros([len(least_intervention), ])
for ii in range(len(least_intervention)):
    #print(ii)
    x, y = find_max(least_intervention[ii])
    bp_s[ii] = x
    bp_i[ii] = y

#%%
fig, ax = plt.subplots()
#ax.set_xlim(A.sbar, 1)
#ax.set_ylim(0, A.imax*1.1)
#ax.plot(least_intervention[75].s, least_intervention[75].i)
ax.plot(s0, s0)
ax.plot(s0, bp_s, "b-")
ax.plot(s0, final_point_i, "r-")


#%%
fig, ax = plt.subplots()
ax.set_xlabel("Idx")
ax.set_ylabel(r"$\frac{T_c}{T_t}$")
ax.plot(A.points[0].i_times / A.points[0].times, "b-")
ax.plot(A.points[11].i_times / A.points[11].times, "r-")
ax.plot(A.points[23].i_times / A.points[23].times, "g-")
ax.plot(A.points[35].i_times / A.points[35].times, "y-")
ax.plot(A.points[47].i_times / A.points[47].times, "k-")
ax.plot(A.points[59].i_times / A.points[59].times, "c-")
