#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:55:50 2020

@author: rmm
"""

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import matplotlib.pyplot as plt
import sir as SIR
import pickle
import os
import csv

def u0(state, sys):
    return 0

def umax(state, sys):
    return sys.umax

def ds(t, state, u_func, evol, crit):
    s = state[0]
    i = state[1]
    beta = evol.beta
    u = u_func(state, crit)
    #print(u)
    return -(1-u)*beta*s*i

def di(t, state, u_func, evol, crit):
    s = state[0]
    i = state[1]
    beta = evol.beta
    gamma = evol.gamma
    u = u_func(state, crit)
    #print(u)
    return (1-u)*beta*s*i - gamma*i

def sir(t, state, u_func, evol, crit, term_val):
    """
    A
    """
    #print(state)
    dF = np.array([ds(t, state, u_func, evol, crit),
                   di(t, state, u_func, evol, crit)]).reshape([2, ])
    return dF

def u_crit(state, sys):
    """
    A FUNCTION THAT WILL DETERMINE THE VALUE OF THE CONTROL DEPENDING ON THE
    POSITION IN A SYSTEM.
    """
    s = state[0]
    i = state[1]
    tau = scipy.interpolate.interp1d(sys.tau.s, sys.tau.i, kind = "cubic")
    phi = scipy.interpolate.interp1d(sys.phi.s, sys.phi.i, kind = "cubic")
    cc = scipy.interpolate.interp1d(sys.commutation_curve[0],
                                    sys.commutation_curve[1],
                                    kind = "cubic")
    if i > sys.imax:
        return sys.umax
    if s <= sys.commutation_curve[0][-1]:
        #print("Case 1")
        if s < sys.sbar or i < tau(s):
            return 0
        return sys.umax
    elif s > sys.commutation_curve[0][-1] and s < sys.commutation_curve[0][0]:
        #print("Case 2")
        if ((i > tau(s)) and (i < cc(s))) or (i > sys.imax):
            return sys.umax
        elif i > cc(s) and i < sys.imax:
            return 0
        else:
            return 0
    else:
        #print("Case 3")
        if i > sys.imax:
            return sys.umax
        elif s > sys.sstar and i > phi(s):
            return sys.umax
        return 0


def u_opt(state, sys, e = 1e-8):
    s = state[0]
    i = state[1]
    dom = (sys.commutation_curve[0][-1], sys.commutation_curve[0][0])
    if i < tau(state, sys):
        #print("Did cond. 1.")
        return 0
    elif i > phi(state, sys):
        if s > dom[1]:
            #print("Did cond. 2.")
            return reg(i - phi(state, sys), sys.umax, e)
        else:
            #print("Did cond. 3.")
            return sys.umax
    else:
        if s < dom[0]:
            #print("Did cond. 4.")
            return sys.umax
        elif s > dom[1]:
            #print("Did cond. 5.")
            return reg(i - phi(state, sys), sys.umax, e)
        else:
            if i > cc(state, sys):
                #print("Did cond. 6.")
                return 0
            else:
                #print("Did cond. 7.")
                return sys.umax


def reg(x, um, e):
    m = um / (2*e)
    b = um / 2
    return np.clip(m*x + b, 0, um)


def phi(state, sys):
    S, I = state
    foo = scipy.interpolate.interp1d(sys.phi.s, sys.phi.i, kind = "cubic")
    if S < sys.sstar:
        return sys.imax
    else:
        return foo(S)

def tau(state, sys):
    S, I = state
    foo = scipy.interpolate.interp1d(sys.tau.s, sys.tau.i, kind = "cubic")
    if S < sys.sbar:
        return sys.imax
    else:
        return foo(S)

def cc(state, sys):
    S, I = state
    foo = scipy.interpolate.interp1d(sys.commutation_curve[0],
                                     sys.commutation_curve[1],
                                     kind = "cubic")
    return foo(S)


def final_cond(t, state, u_func, evol, crit, term_val):
    out = state[0] - term_val
    return out

final_cond.terminal = True

#%%
i = 1e-4
s = 1 - i
x0 = [s, i]

ireal = 0.1
ureal = 0.6
rreal = 3.64
Re = SIR.SIR()
Re.set_params([ireal, ureal, rreal], flag = "r")
print("Setting up the real system.")
SIR.find_commutation_curve(Re)

#%%
iest = 0.1
uest = 0.6
quo = np.linspace(0.6, 1.4, 15, endpoint = True)
r_vals = quo * rreal
overshoot = np.zeros(np.shape(r_vals))
sols = np.empty(np.shape(r_vals), dtype = object)
mods = np.empty(np.shape(r_vals), dtype = object)
for ii in range(np.shape(r_vals)[0]):
    rest = r_vals[ii]
    print(rest)
    Est = SIR.SIR()
    Est.set_params([iest, uest, rest], flag = "r")
    print("Setting up one of the estimated systems.")
    SIR.find_commutation_curve(Est)
    
    print("Simulating...")
    sol = scipy.integrate.solve_ivp(sir, [0, 10000], x0,
                                    args = (u_opt, Re, Est, 1 / max(r_vals)),
                                    #method = "LSODA", min_step = 1e-3,
                                    method = "RK23",
                                    events = [final_cond])
    sols[ii] = sol
    mods[ii] = Est
    overshoot[ii] = max(sol.y[1])

f = 2000
dur = 1
os.system("play -nq -t alsa synth {} sine {}".format(dur, f))

#f = open("robustness.pckl", "wb")
#pickle.dump([Re, r_vals, quo, overshoot, sols, mods], f)
#f.close()

#%%
#f = open("robustness.pckl", "rb")
#Re, r_vals, quo, overshoot, sols, mods = pickle.load(f)
#f.close()

fig, ax = plt.subplots(figsize = (18, 12))
ax.plot(quo, (overshoot - Re.imax) / Re.imax, "b.")
ax.set_xlabel(r"$\dfrac{R_e}{R_r}$")
ax.set_ylabel(r"$\dfrac{max(I) - I_{max}}{I_{max}}$")
#fig.savefig("docs/robustness/robustness_364_06.pdf", format = "pdf")
#fig.savefig("docs/robustness/robustness_364_06.jpg", format = "jpg")

fig, ax = plt.subplots(figsize = (18, 12))
ax.set_xlim(mods[-1].sbar, 1)
ax.set_ylim(0, Re.imax*2)
ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$I$")
ax.plot(Re.tau.s, Re.tau.i, "b:")
ax.plot(mods[0].tau.s, mods[0].tau.i, "b--")
ax.plot(mods[-1].tau.s, mods[-1].tau.i, "b-.")
ax.plot(Re.phi.s, Re.phi.i, "r:")
ax.plot(mods[0].phi.s, mods[0].phi.i, "r--")
ax.plot(mods[-1].phi.s, mods[-1].phi.i, "r-.")
ax.plot(mods[0].commutation_curve[0], mods[0].commutation_curve[1], "b--")
ax.plot(mods[-1].commutation_curve[0], mods[-1].commutation_curve[1], "b-.")
ax.plot(sols[0].y[0], sols[0].y[1])
#ax.plot(sols[1].y[0], sols[1].y[1])
#ax.plot(sols[2].y[0], sols[2].y[1])
#ax.plot(sols[3].y[0], sols[3].y[1])
#ax.plot(sols[4].y[0], sols[4].y[1])
#ax.plot(sols[5].y[0], sols[5].y[1])
#ax.plot(sols[6].y[0], sols[6].y[1])
#ax.plot(sols[7].y[0], sols[7].y[1])
#ax.plot(sols[8].y[0], sols[8].y[1])
#ax.plot(sols[9].y[0], sols[9].y[1])
#ax.plot(sols[10].y[0], sols[10].y[1])
#ax.plot(sols[11].y[0], sols[11].y[1])
#ax.plot(sols[12].y[0], sols[12].y[1])
#ax.plot(sols[13].y[0], sols[13].y[1])
#ax.plot(sols[14].y[0], sols[14].y[1])
ax.plot([0, 1], [Re.imax, Re.imax], "k:")
ax.legend(["Parámetros Reales", "Primer Estimado", "Último Estimado"])
#fig.savefig("docs/robustness/robustness_check_364_06.pdf", format = "pdf")
#fig.savefig("docs/robustness/robustness_check_364_06.jpg", format = "jpg")

#%%
it_from_s = scipy.interpolate.interp1d(sol.y[0], [sol.y[1], sol.t],
                                       kind = "cubic")
print(it_from_s(Re.sbar))
