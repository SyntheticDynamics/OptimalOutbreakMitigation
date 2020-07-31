# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 17:05:46 2020

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

def make_initial_conditions(system, displacement, low):
    """
    A FUNCTION THAT CREATES THE INITIAL CONDITIONS FOR THE SEARCH OF THE
    COMMUTATION CURVE IN A DETERMINED SYSTEM. TAKES AS INPUT THE SYSTEM THAT
    IS GOING TO BE ANALYZED, THE displacement TO THE RIGHT, AND HOW FAR low THE
    LINE SEGMENT IS GOING TO GO.
    """
    C = sir.CurveSegment(system.sbar, system.imax, 0, system, 1)
    C.s = C.s + displacement
    s_inter, i_inter = C._curve_sol(low)
    C = sir.CurveSegment(system.sbar, system.imax, 0, system, s_inter)
    C.s = C.s + displacement
    s0 = np.linspace(C.s[-1], 1)
    i0 = np.array([i_inter]*len(s0))
    s0 = np.concatenate((C.s, s0))
    i0 = np.concatenate((C.i, i0))
    return s0, i0

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
A.set_params([imax, umax, gamma, beta], flag = "bg")
A._find_curves()

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
ax.plot(s0, i0, "b:")
ax.legend([r"$\tau$", r"$\phi$", r"$\theta$", r"$\rho$", "SP"])
#fig.show()

#%%
for ii in range(len(s0)):
    A.add_point(s0[ii], i0[ii])
A.find_regions()
A.get_shortest()

M = np.array([p.least_time for p in A.points])

fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax*1.1)
ax.plot(A.tau.s, A.tau.i, "r-")
ax.plot(A.phi.s, A.phi.i, "b-")
for ii in range(np.shape(M)[0])[::int(np.shape(M)[0] / 10)]:
    ax.plot(M[ii].s, M[ii].i, "g-")
#fig.show()

#%%
# CREATE AN ARRAY OF ALL THE LAST POINTS (IN S) OF THE SHORTEST TRAJECTORIES.
final_point = np.array([tra.s[-1] for tra in M])
fig, ax = plt.subplots()
ax.plot(s0, final_point)
#fig.show()

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
#fig.show()

A = sir.SIR()
A.set_params([imax, umax, gamma, beta], flag = "bg")
A._find_curves()

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
fig.show()

s_inter, i_inter = C._curve_sol(i_low)
C = sir.CurveSegment(A.sbar, A.imax, 0, A, s_inter)
C = sir.CurveSegment(s0[crit1]-displacement,
                     i0[crit1], 0, A, s_inter, 100)
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

s0 = np.linspace(C.s[-1], s0[crit2], 100)
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
#fig.show()

#%%
for ii in range(len(s0)):
    A.add_point(s0[ii], i0[ii])
A.find_regions()
A.get_shortest()

M = np.array([p.least_time for p in A.points])

fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax*1.1)
ax.plot(A.tau.s, A.tau.i, "r-")
ax.plot(A.phi.s, A.phi.i, "b-")
for ii in range(np.shape(M)[0])[::int(np.shape(M)[0] / 10)]:
    ax.plot(M[ii].s, M[ii].i, "g-")
#fig.show()

#%%
final_point = np.array([tra.s[-1] for tra in M])
fig, ax = plt.subplots()
ax.plot(s0, final_point)
#fig.show()

#%%
bp_s = np.zeros([len(M), ])
bp_i = np.zeros([len(M), ])

for ii in range(len(M)):
    x, y = find_max(M[ii])
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
#fig.show()


#%%
phi_max = np.where(A.phi.i == A.imax)[0][0]

fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax * 1.1)
ax.fill_between([A.sbar, 1], [A.imax]*2, [1, 1],
                color = "xkcd:red", alpha = 0.3)
ax.fill_between([A.sbar, 1], [A.imax]*2,
                color = "xkcd:blue", alpha = 0.3)
ax.fill_between(A.phi.s[:phi_max], A.phi.i[:phi_max], [A.imax]*phi_max,
                color = "xkcd:white", alpha = 1)
ax.fill_between(A.phi.s[:phi_max], A.phi.i[:phi_max], [A.imax]*phi_max,
                color = "xkcd:red", alpha = 0.3)
#ax.fill_between(A.tau.s[::50], A.tau.i[::50], bp_s,
#                color = "xkcd:white", alpha = 1)
ax.plot(A.tau.s, A.tau.i, "b-")
ax.plot(A.phi.s, A.phi.i, "r-")
ax.plot([0.4, 1], [A.imax, A.imax], "k:")
ax.plot(bp_s, bp_i, "k-")
