# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:08:09 2020

@author: Rodrigo
"""

import sir
import matplotlib.pyplot as plt
import numpy as np

#%%
"""
A FEW FUNCTIONS THAT WILL BE USED FURTHER DOWN.
"""
def find_max(trajectory):
    """
    FINDS THE COORDINATE OF THE MAXIMUM POINT IN A TRAJECTORY.
    """
    x = trajectory.s
    y = trajectory.i
    yt = np.abs(y - max(y))
    yt = yt < 1e-5
    max_idx = np.where(yt == True)[0]
    max_idx = max(max_idx)
    return [x[max_idx], y[max_idx]]

def find_relevant_change(array, err = 1e-3):
    """
    FIND THE POINT AT WHICH AN ARRAY STARTS CHANGING "SIGNIFICANTLY".
    """
    dif = np.abs(array - array[0])
    dif = dif > err
    idx = min(np.where(dif == True)[0])
    return idx
    

#%%
imax = 0.1
umax = 0.5
gamma = 0.2
beta = 0.5

# SET HOW FAR TO THE RIGHT THE REFERENCE CURVE BASED ON SIR.tau WILL BE MOVED.
displacement = 0.001
# SET HOW FAR LOW THE CURVE SIR.tau WILL GO.
i_low = 0.0005

#%%
# CREATE SYSTEM, SET PARAMETERS, AND FIND CURVES tau, phi ET CETERA.
A = sir.SIR()
A.set_params(imax, umax, gamma, beta)
A.find_curves()

# CREATE A CURVE THAT IS A CLONE OF A.tau AND DISPLACE IT TO THE RIGHT BY 
# displacement UNITS. USE THIS CURVE AND A LINSPACE AT I = i_low TO CREATE ALL
# INITIAL CONDITIONS, s0 AND i0.
C = sir.CurveSegment(A.sbar, A.imax, 0, A, 1)
C.s = C.s + displacement
s_inter, i_inter = C._curve_sol(i_low)
C = sir.CurveSegment(A.sbar, A.imax, 0, A, s_inter)
C.s = C.s + displacement
s0 = np.linspace(C.s[-1], 1)
i0 = np.array([i_inter]*len(s0))
s0 = np.concatenate((C.s, s0))
i0 = np.concatenate((C.i, i0))

# A PLOT TO SEE IF ALL IS ALRIGHT.
fig, ax = plt.subplots()
ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$I$")
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax*1.1)
ax.set_title("System Curves")
ax.plot(A.tau.s, A.tau.i, "r-")
ax.plot(A.phi.s, A.phi.i, "b-")
ax.plot(A.theta.s, A.theta.i, "g-")
ax.plot(A.rho.s, A.rho.i, "k-")
ax.legend([r"$\tau$", r"$\phi$", r"$\theta$", r"$\rho$"])
#ax.plot(s0, i0, "bx")

#%%
# CREATE POINTS ON A FOR ALL INITIAL CONDITIONS, FIND THEIR REGIONS, GET THEIR
# TRAJECTORIES AND THE SHORTEST-TIME TRAJECTORY.
for ii in range(len(s0)):
    A.add_point(s0[ii], i0[ii])
A.find_regions()
A.get_trajectories()

# CALCULATE TIME, GET SHORTEST TRAJECTORIES FOR EACH POINT AND STORE THEM IN AN
# ARRAY.
least_time = np.empty([len(A.points), ], dtype = object)
for ii in range(len(A.points)):
    p = A.points[ii]
    p.get_times()
    least_time[ii] = p.get_least_time()

# CREATE AN ARRAY OF ALL THE LAST POINTS (IN S) OF THE SHORTEST TRAJECTORIES.
final_point = np.array([tra.s[-1] for tra in least_time])

fig, ax = plt.subplots()
ax.set_xlabel(r"$s_0$")
ax.set_ylabel(r"$s_f$")
ax.set_title("Initial and last points of least-time trajectories")
ax.plot(s0, final_point)

fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax*1.1)
ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$I$")
ax.set_title("Sample of least-time trajectories")
ax.plot(A.tau.s, A.tau.i, "b--")
ax.plot([0, 1], [A.imax]*2, "r--")
ax.plot(least_time[46].s, least_time[46].i)
ax.plot(least_time[47].s, least_time[47].i)
ax.plot(least_time[48].s, least_time[48].i)
ax.plot(least_time[49].s, least_time[49].i)
ax.plot(least_time[59].s, least_time[59].i)
ax.plot(least_time[69].s, least_time[69].i)
ax.plot(least_time[79].s, least_time[79].i)
ax.plot(least_time[89].s, least_time[89].i)
ax.plot(least_time[99].s, least_time[99].i)
ax.legend([r"$\tau$", r"$I_{max}$"])

#%%
fig, ax = plt.subplots()
ax.set_xlim(0.7, 0.8)
ax.set_ylim(0, 0.02)
ax.plot(A.tau.s, A.tau.i, "r-")
ax.plot(least_time[43].s, least_time[43].i)
ax.plot(least_time[44].s, least_time[44].i)
ax.plot(least_time[45].s, least_time[45].i)
ax.plot(least_time[46].s, least_time[46].i)
ax.plot(least_time[47].s, least_time[47].i)
ax.plot(least_time[48].s, least_time[48].i)
ax.plot(least_time[49].s, least_time[49].i)
ax.legend(["tau", 1,2,3,4,5,6,7])


#%%
"""
FIND START AND END OF THE DECLINE IN s_0 vs s_f.
"""
fp_diff = np.diff(final_point)
crit1 = find_relevant_change(fp_diff)
max_idx = max(np.where(final_point == max(final_point))[0])
st = s0[crit1]
crit2 = min(np.where(fp_diff[max_idx:] > 0)[0]) + max_idx

fig, ax = plt.subplots()
ax.plot(s0[crit1], final_point[crit1], "bx")
ax.plot(s0[max_idx], final_point[max_idx], "r+")
ax.plot(s0[crit2], final_point[crit2], "rx")
ax.plot(s0, final_point)
ax.legend(["Start", "Max", "End"])


#%%
"""
DO IT ALL OVER AGAIN, BUT NOW ONLY IN THE REGIONS OF INTEREST, DETERMINED BY 
crit1 AND crit2.
"""
A = sir.SIR()
A.set_params(imax, umax, gamma, beta)
A.find_curves()

C = sir.CurveSegment(A.sbar, A.imax, 0, A, 1)
C.s = C.s + displacement

fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax*1.1)
ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$I$")
ax.plot(A.tau.s, A.tau.i, "r:")
ax.plot(C.s, C.i, "b-")
ax.plot(s0[crit1], i0[crit1], "rx")
ax.legend([r"$\tau$", "Reference Curve", "Start"])

s_inter, i_inter = C._curve_sol(i_low)
C = sir.CurveSegment(A.sbar, A.imax, 0, A, s_inter)
C = sir.CurveSegment(s0[crit1]-displacement,
                     i0[crit1], 0, A, s_inter, 1000)
C.s = C.s + displacement

"""
fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, 0.002)
ax.plot(A.tau.s, A.tau.i, "r-")
ax.plot(C.s, C.i, "b.")
ax.plot(s0[crit1], i0[crit1], "rx")
ax.set_title("2")
"""

s0 = np.linspace(C.s[-1], s0[crit2], 1000)
i0 = np.array([i_inter]*len(s0))
s0 = np.concatenate((C.s, s0))
i0 = np.concatenate((C.i, i0))

fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax*1.1)
ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$I$")
ax.plot(A.tau.s, A.tau.i, "r-")
ax.plot(s0, i0, "b:")
ax.set_title("3")
ax.legend([r"$\tau$", "Domain of study"])
ax.set_title("Initial Conditions")

#%%
for ii in range(len(s0)):
    A.add_point(s0[ii], i0[ii])
A.find_regions()
A.get_trajectories()
A.get_shortest()

least_time = np.empty([len(A.points), ], dtype = object)
for ii in range(len(A.points)):
    p = A.points[ii]
    p.get_times()
    least_time[ii] = p.get_least_time()

final_point = [tra.s[-1] for tra in least_time]

fig, ax = plt.subplots()
ax.set_xlabel(r"$s_0$")
ax.set_ylabel(r"$s_f$")
ax.plot(s0, final_point)
ax.set_title("Initial and last points of least-time trajectories")

fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax*1.1)
ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$I$")
ax.plot(A.tau.s, A.tau.i, "b--")
ax.plot([0, 1], [A.imax]*2, "r--")
ax.plot(least_time[0].s, least_time[0].i)
ax.plot(least_time[1000].s, least_time[1000].i)
ax.plot(least_time[1099].s, least_time[1099].i)
ax.plot(least_time[1199].s, least_time[1199].i)
ax.plot(least_time[1299].s, least_time[1299].i)
ax.plot(least_time[1399].s, least_time[1399].i)
ax.plot(least_time[1499].s, least_time[1499].i)
ax.plot(least_time[1599].s, least_time[1599].i)
ax.plot(least_time[1699].s, least_time[1699].i)
ax.plot(least_time[1799].s, least_time[1799].i)
ax.plot(least_time[1899].s, least_time[1899].i)
ax.plot(least_time[1999].s, least_time[1999].i)
ax.legend([r"$\tau$", r"$I_{max}$"])
ax.set_title("Sample of least-time trajectories")

#%%
bp_s = np.zeros([len(least_time), ])
bp_i = np.zeros([len(least_time), ])

for ii in range(len(least_time)):
    x, y = find_max(least_time[ii])
    bp_s[ii] = x
    bp_i[ii] = y

fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax*1.1)
ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$I$")
ax.plot(A.tau.s, A.tau.i, "b-")
ax.plot(A.phi.s, A.phi.i, "r-")
ax.plot(bp_s, bp_i, "g-")
ax.legend([r"$\tau$", r"$\phi$", "Commutation Curve"])
ax.plot([A.sbar, 1], [A.imax]*2, "y--")
h = r"Bp with $u_m$ = {}"
ax.set_title(h.format(umax))
#fig.savefig("docs/bp_{}.pdf".format(umax), format = "pdf")
#fig.savefig("docs/bp_{}.jpg".format(umax), format = "jpg")

#%%
fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax * 1.1)
ax.fill_between(A.tau.s, A.tau.i, color = "C0")
ax.fill_between(A.phi.s, A.phi.i, color = "C0")
ax.fill_between(A.phi.s, A.phi.i, [A.imax*1.1]*len(A.tau.i), color = "C1")
#ax.fill_between(A.tau.s, A.tau.i, [A.imax*1.1]*len(A.tau.i), color = "C1")
#ax.fill_between(bp_s, bp_i, [A.imax]*len(bp_s), color = "C0")
#ax.fill_between(bp_s, A.phi.s[::5], color = "C0")
ax.plot(bp_s, bp_i, "r-")
