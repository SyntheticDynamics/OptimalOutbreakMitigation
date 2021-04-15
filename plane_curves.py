#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 09:29:56 2021

@author: rmm
"""

import numpy as np
import sir
import scipy.interpolate
import matplotlib.pyplot as plt

#%%
def plot_curves(system):
    if system.commutation_curve is None:
        print("There is no commutation curve calculated yet.")
        sir.find_commutation_curve(system)
    fig, ax = plt.subplots()
    ax.set_xlim(system.sbar, 1)
    ax.set_ylim(0, system.imax * 1.1)
    
    s = list(np.linspace(0, 1))
    T = [tau(si, system) for si in s]
    P = [phi(si, system) for si in s]
    C = [com_curve(si, system) for si in s]
    
    ax.fill_between(s, T, 0, facecolor = "blue", alpha = 0.5)
    ax.fill_between(s, C, P, facecolor = "blue", alpha = 0.5)
    ax.fill_between(s, P, 1, facecolor = "red", alpha = 0.5)
    ax.fill_between(s, T, C, facecolor = "red", alpha = 0.5)
    return fig


def tau(s, sys):
    tt = scipy.interpolate.interp1d(sys.tau.s, sys.tau.i, kind = "cubic")
    if s < sys.sbar:
        return sys.imax
    elif s >= sys.sbar and s < sys.tau._curve_sol()[0]:
        return tt(s)
    else:
        return 0


def phi(s, sys):
    pp = scipy.interpolate.interp1d(sys.phi.s, sys.phi.i, kind = "cubic")
    if s < sys.sstar:
        return sys.imax
    elif s >= sys.sstar and s < sys.phi._curve_sol()[0]:
        return pp(s)
    else:
        return 0


def com_curve(s, sys):
    cc = scipy.interpolate.interp1d(sys.commutation_curve[0],
                                    sys.commutation_curve[1],
                                    kind = "cubic")
    if s < sys.commutation_curve[0][0]:
        return sys.imax
    elif s >= sys.commutation_curve[0][0] and s < sys.commutation_curve[0][-1]:
        return cc(s)
    else:
        return 0

#%%
imax = 0.1
umax = 0.5
beta = 0.5
gamma = 0.2

A = sir.SIR()
A.set_params([imax, umax, gamma, beta], flag = "bg")
sir.find_commutation_curve(A)

#%%
plot_curves(A)

#%%
"""
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, A.imax*1.1)
ax.plot(A.phi.s, A.phi.i, "b-")
ax.plot(A.tau.s, A.tau.i, "r-")
ax.plot(A.theta.s, A.theta.i, "k-")
ax.plot(A.rho.s, A.rho.i, "y-")
ax.plot(A.commutation_curve[0], A.commutation_curve[1], "g-")
ax.legend(["phi", "tau", "theta", "rho", "cc"])
ax.plot([0, 1], [A.imax]*2, "k:")
"""
