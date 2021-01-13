#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:37:52 2020

@author: rmm
"""

import sir
import scipy.integrate
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def dS(t, state, u_func, evolution, control, termination):
    S, I, R = state
    beta = evolution.beta
    u = u_func(state, control)
    return -(1 - u) * beta * S * I


def dI(t, state, u_func, evolution, control, termination):
    S, I, R = state
    beta = evolution.beta
    gamma = evolution.gamma
    u = u_func(state, control)
    return (1 - u) * beta * S * I - gamma * I


def dR(t, state, u_func, evolution, control, termination):
    S, I, R = state
    gamma = evolution.gamma
    return gamma * I


def F(t, state, u_func, evolution, control, termination):
    S = dS(t, state, u_func, evolution, control, termination)
    I = dI(t, state, u_func, evolution, control, termination)
    R = dR(t, state, u_func, evolution, control, termination)
    return np.array([S, I, R]).reshape([3, ])


def u_opt(state, sys, e = 1e-10):
    S, I, R = state
    dom = (sys.commutation_curve[0][-1], sys.commutation_curve[0][0])
    if I < tau(state, sys):
        #print("Did cond. 1.")
        return 0
    elif I > phi(state, sys):
        if S > dom[1]:
            #print("Did cond. 2.")
            return reg(I - phi(state, sys), sys.umax, e)
        else:
            #print("Did cond. 3.")
            return sys.umax
    else:
        if S < dom[0]:
            #print("Did cond. 4.")
            return sys.umax
        elif S > dom[1]:
            #print("Did cond. 5.")
            return reg(I - phi(state, sys), sys.umax, e)
        else:
            if I > cc(state, sys):
                #print("Did cond. 6.")
                return 0
            else:
                #print("Did cond. 7.")
                return sys.umax


def u_0(state, sys, e = 1e-10):
    return 0


def u_max(state, sys, e = 1e-10):
    return sys.umax


def reg(x, um, e):
    m = um / (2*e)
    b = um / 2
    return np.clip(m*x + b, 0, um)


def phi(state, sys):
    S, I, R = state
    foo = scipy.interpolate.interp1d(sys.phi.s, sys.phi.i, kind = "cubic")
    if S < sys.sstar:
        return sys.imax
    else:
        return foo(S)


def tau(state, sys):
    S, I, R = state
    foo = scipy.interpolate.interp1d(sys.tau.s, sys.tau.i, kind = "cubic")
    if S < sys.sbar:
        return sys.imax
    else:
        return foo(S)


def cc(state, sys):
    S, I, R = state
    foo = scipy.interpolate.interp1d(sys.commutation_curve[0],
                                     sys.commutation_curve[1],
                                     kind = "cubic")
    return foo(S)


def final_cond(t, state, u_func, evol, crit, term_val):
    out = state[0] - term_val
    return out


final_cond.terminal = True

#%%
x0 = np.array([1 - 1e-4, 1e-4, 0]).reshape([3, ])

imax = 0.1
umax = 0.6
R_reals = np.array([1.73, 2.5, 3.64])
ratios = np.linspace(0.7, 1.3, 15)
overshoot = np.zeros([np.shape(R_reals)[0], np.shape(ratios)[0]])
sols = np.empty(np.shape(overshoot), dtype = object)
for ii in range(np.shape(R_reals)[0]):
    rreal = R_reals[ii]
    print("\nR_0 = {}".format(rreal))
    print("Setting up real system.")
    Dyn = sir.SIR()
    Dyn.set_params([imax, umax, rreal], flag = "r")
    sir.find_commutation_curve(Dyn)
    
    for jj in range(np.shape(ratios)[0]):
        restim = rreal * ratios[jj]
        print("\tR_e = {}".format(restim))
        print("Setting up estimated system.")
        Ctrol = sir.SIR()
        Ctrol.set_params([imax, umax, restim], flag = "r")
        sir.find_commutation_curve(Ctrol)
        
        sol = scipy.integrate.solve_ivp(F, [0, 1e5], x0,
                                        args = (u_opt, Dyn, Ctrol, Dyn.sbar),
                                        method = "RK23", events = [final_cond],
                                        rtol = 1e-4, atol = 1e-7)
        S, I, R = sol.y
        over = max(I)
        overshoot[ii, jj] = over
        
        sols[ii, jj] = sol
        
        total = np.shape(overshoot)[0] * np.shape(overshoot)[1]
        current = (jj + 1) + (ii * np.shape(R_reals)[0])
        percentage = current / total
        print("{}% done.".format(percentage*100))

f = 2000
dur = 1
os.system("play -nq -t alsa synth {} sine {}".format(dur, f))

#%%
metric = (overshoot - imax) / imax
fig, ax = plt.subplots(figsize = (18, 12))
ax.plot(ratios, metric[0,:], "g-")
ax.plot(ratios, metric[1,:], "b-")
ax.plot(ratios, metric[2,:], "k-")
ax.plot([0, 2], [0, 0], "b-", alpha = 0.3)
ax.plot([1, 1], [-1, 1], "b-", alpha = 0.3)
ax.set_xlim(0.7, 1.3)
ax.set_ylim(-0.4, 0.2)
ax.legend(R_reals)
ax.set_xlabel(r"$\dfrac{\hat{R}_0}{R_0}$")
ax.set_ylabel(r"$\dfrac{max(I) - I_{max}}{I_{max}}$")
fig.savefig("docs/robustness/ala_me.jpg", format = "jpg")

fig, ax = plt.subplots(3, 1, figsize = (18, 36))
ax[0].set_xlim(0, 1)
ax[1].set_xlim(0, 1)
ax[2].set_xlim(0, 1)
ax[0].set_ylim(0, imax * 2)
ax[1].set_ylim(0, imax * 2)
ax[2].set_ylim(0, imax * 2)
ax[2].set_xlabel(r"$S$")
ax[1].set_ylabel(r"$I$")

ax[0].plot(sols[0, 0].y[0], sols[0, 0].y[1], "b-")
ax[0].plot(sols[0, 3].y[0], sols[0, 3].y[1], "r-")
ax[0].plot(sols[0, 7].y[0], sols[0, 7].y[1], "g-")
ax[0].plot(sols[0, 11].y[0], sols[0, 11].y[1], "y-")
ax[0].plot(sols[0, 14].y[0], sols[0, 14].y[1], "k-")
ax[0].set_title(r"$R_{{0}}$ = {}".format(R_reals[0]))
ax[0].legend(ratios[[0, 3, 7, 11, 14]])

ax[1].plot(sols[1, 0].y[0], sols[1, 0].y[1], "b-")
ax[1].plot(sols[1, 3].y[0], sols[1, 3].y[1], "r-")
ax[1].plot(sols[1, 7].y[0], sols[1, 7].y[1], "g-")
ax[1].plot(sols[1, 11].y[0], sols[1, 11].y[1], "y-")
ax[1].plot(sols[1, 14].y[0], sols[1, 14].y[1], "k-")
ax[1].set_title(r"$R_{{0}}$ = {}".format(R_reals[1]))
ax[1].legend(ratios[[0, 3, 7, 11, 14]])

ax[2].plot(sols[2, 0].y[0], sols[2, 0].y[1], "b-")
ax[2].plot(sols[2, 3].y[0], sols[2, 3].y[1], "r-")
ax[2].plot(sols[2, 7].y[0], sols[2, 7].y[1], "g-")
ax[2].plot(sols[2, 11].y[0], sols[2, 11].y[1], "y-")
ax[2].plot(sols[2, 14].y[0], sols[2, 14].y[1], "k-")
ax[2].set_title(r"$R_{{0}}$ = {}".format(R_reals[2]))
ax[2].legend(ratios[[0, 3, 7, 11, 14]])
fig.savefig("docs/robustness/ala_me_check.jpg", format = "jpg")

#%%
with open("robustez_sir.csv", "w") as f:
    pt = csv.writer(f, delimiter = ",")
    pt.writerow(ratios)
    pt.writerow(metric[0,:])
    pt.writerow(metric[1,:])
    pt.writerow(metric[2,:])
