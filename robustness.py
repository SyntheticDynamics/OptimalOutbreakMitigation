# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:42:32 2020

@author: Rodrigo
"""

import sir
import numpy as np
import matplotlib.pyplot as plt

#%%
R0 = [2.1, 2.5] # [minR0, maxR0]
imax = 0.1
umax = 0.5

disp = 1e-3
ilow = 5e-4

#%%
scenarios = {"bcs": None, "wcs": None}
for ii in range(len(R0)):
    r0 = R0[ii]
    key = list(scenarios.keys())[ii]
    A = sir.SIR()
    A.set_params([imax, umax, r0], flag = "r")
    scenarios[key] = A
    A._find_curves()
    s0, i0 = sir.create_initial_conditions(A, disp, ilow)
    
    for ii, jj in zip(s0, i0):
        A.add_point(ii, jj)
    
    A.find_regions()
    A.get_shortest()
    M = np.array([p.least_time for p in A.points])
    
    final_point = np.array([tra.s[-1] for tra in M])
    fig, ax = plt.subplots()
    ax.plot(s0, final_point)
    
    """
    fp_diff = np.diff(final_point)
    crit1 = sir.find_relevant_change(fp_diff)
    max_idx = max(np.where(final_point == max(final_point))[0])
    st = s0[crit1]
    print(st)
    crit2 = min(np.where(fp_diff[max_idx:] > 0)[0]) + max_idx
    end = s0[crit2]
    print(end)
    """
    st, end = sir.find_criticals(s0, final_point)
    
    s0, i0 = sir.create_initial_conditions(A, disp, ilow, st, end, size = 100)
    
    A.remove_all_points()
    for ii, jj in zip(s0, i0):
        A.add_point(ii, jj)
    
    A.find_regions()
    A.get_shortest()
    M = np.array([p.least_time for p in A.points])
    
    bp_s = np.zeros([len(M), ])
    bp_i = np.zeros([len(M), ])

    for ii in range(len(M)):
        x, y = sir.find_max(M[ii])
        bp_s[ii] = x
        bp_i[ii] = y
    
    A.commutation_curve = [bp_s, bp_i]


#%%
A = sir.SIR()
A.set_params([0.1, 0.5, 2.4], flag = "r")
P = A.add_point(0.85, 0.01)
P2 = A.add_point(0.95, 0.01)
A._find_curves()
A.find_regions()
A.get_shortest()
T = P.least_time
T2 = P2.least_time




wcs = scenarios["wcs"]
bcs = scenarios["bcs"]

fig, ax = plt.subplots()
ax.set_xlim(wcs.sbar, 1)
ax.set_xlabel(r"$S$")
ax.set_ylim(0, imax * 1.1)
ax.set_ylabel(r"$I$")
ax.plot(wcs.tau.s, wcs.tau.i, "b--")
ax.plot(bcs.tau.s, bcs.tau.i, "b-")
ax.plot(T.s, T.i, color = "xkcd:vomit green", linestyle = "-.")
ax.plot(T2.s, T2.i, color = "xkcd:vomit green", linestyle = "-.")
ax.plot(wcs.phi.s, wcs.phi.i, "r--")
ax.plot(bcs.phi.s, bcs.phi.i, "r-")
#ax.plot(wcs.rho.s, wcs.rho.i, "g--")
#ax.plot(bcs.rho.s, bcs.rho.i, "g-")
#ax.plot(wcs.theta.s, wcs.theta.i, "k--")
#ax.plot(bcs.theta.s, bcs.theta.i, "k-")
ax.plot(wcs.commutation_curve[0], wcs.commutation_curve[1], "k--")
ax.plot(bcs.commutation_curve[0], bcs.commutation_curve[1], "k-")
ax.plot([0, 1], [imax]*2, "k:")
ax.legend([r"$R_0 = 2.5$", r"$R_0 = 2.1$", "Trajectory"])
ax.set_title(("Best and Worst case scenarios, and trajectories with\n"
              "intermediate value"))
#fig.savefig("docs/wcs-bcs.jpg", format = "jpg")
#fig.savefig("docs/wcs-bcs.pdf", format = "pdf")

#%%
s1, i1 = [0.85, 0.01]
s2, i2 = [0.95, 0.01]
r0 = np.linspace(2.1, 2.5)
ni1 = np.zeros(np.shape(r0), dtype = float)
ni2 = np.zeros(np.shape(r0), dtype = float)

for ii in range(np.shape(r0)[0]):
    r = r0[ii]
    B = sir.SIR()
    B.set_params([imax, umax, r], flag = "r")
    B._find_curves()
    P1 = B.add_point(s1, i1)
    P2 = B.add_point(s2, i2)
    B.find_regions()
    B.get_shortest()
    M1 = P1.least_time
    M2 = P2.least_time
    tim1 = max(M1.i)
    tim2 = max(M2.i)
    ni1[ii] = (imax - tim1) / imax
    ni2[ii] = (imax - tim2) / imax

fig, ax = plt.subplots()
ax.set_xlabel(r"$R_0$")
ax.set_ylim(0, 1)
ax.set_ylabel(r"$\dfrac{I_{max} - max(I)}{I_{max}}$")
ax.plot(r0, ni1)
ax.plot(r0, ni2)
ax.legend(["(0.85, 0.01)", "(0.95, 0.01)"])
#fig.savefig("docs/r0_imax.jpg", format = "jpg")
#fig.savefig("docs/r0_imax.pdf", format = "pdf")
