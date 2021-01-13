#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:02:29 2020

@author: rmm
"""

import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import sir
import os
import csv

class ref():
    """
    AN OBJECT THAT WILL HAVE JUST A SET OF PARAMETERS AS OBJECT ATTRIBUTES FOR
    REASONS.
    """
    def __init__(self, beta, gamma, lamb, p):
        self.beta = beta
        self.gamma = gamma
        self.lamb = lamb
        self.p = p



def dS(t, state, u_func, ref, crit):
    S, E, Is, Ia, R = state
    beta = ref.beta
    u = u_func(state, crit)
    #print(u)
    return -(1 - u) * beta * S * (Ia + Is)

def dIs(t, state, u_func, ref, crit):
    S, E, Is, Ia, R = state
    p = ref.p
    gamma = ref.gamma
    lamb = ref.lamb
    return p * lamb * E - gamma * Is

def dIa(t, state, u_func, ref, crit):
    S, E, Is, Ia, R = state
    gamma = ref.gamma
    p = ref.p
    lamb = ref.lamb
    return (1 - p) * lamb * E - gamma * Ia

def dE(t, state, u_func, ref, crit):
    S, E, Is, Ia, R = state
    u = u_func(state, crit)
    beta = ref.beta
    lamb = ref.lamb
    return (1 - u) * beta * S * (Is + Ia) - lamb * E

def dR(t, state, u_func, ref, crit):
    S, E, Is, Ia, R = state
    gamma = ref.gamma
    return gamma * (Ia + Is)

def SEIIR(t, state, u_func, ref, crit, term_val):
    #print(state)
    S = dS(t, state, u_func, ref, crit)
    E = dE(t, state, u_func, ref, crit)
    Is = dIs(t, state, u_func, ref, crit)
    Ia = dIa(t, state, u_func, ref, crit)
    R = dR(t, state, u_func, ref, crit)
    dSEIIR = np.array([S,
                       E,
                       Is,
                       Ia,
                       R]).reshape([5, ])
    return dSEIIR


def u_0(state, sys):
    return 0


def u_max(state, sys):
    return sys.umax


def u_crit(state, sys):
    """
    A FUNCTION THAT DETERMINES THE CONTROL GIVEN A POINT IN THE PHASE PLANE.
    THIS FUNCTION IS *NOT* REGULARIZED, SO IT WILL BREAK WITH INTEGRATION
    ALGORITHMS THAT DO NOT HAVE A MINIMUM STEP SIZE.
    """
    s = state[0]
    i = state[2]
    dom = (sys.commutation_curve[0][-1], sys.commutation_curve[0][0])
    if i > phi(state, sys):
        return sys.umax
    elif i < tau(state, sys):
        return 0
    else:
        if s < dom[0]:
            return sys.umax
        elif s > dom[1]:
            return 0
        else:
            if i > cc(state, sys):
                return 0
            else:
                return sys.umax


def u_opt(state, sys, e = 1e-3):
    """
    A FUNCTION THAT DETERMINES THE CONTROL FOR A POINT IN THE PLANE. THIS
    FUNCTION *IS* REGULARIZED, SO IT CAND (SUPPOSEDLY) BE USED WITH ANY
    INTEGRATION METHOD.
    """
    s = state[0]
    i = state[2]
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
    S, E, Is, Ia, R = state
    foo = scipy.interpolate.interp1d(sys.phi.s, sys.phi.i, kind = "cubic")
    if S < sys.sstar:
        return sys.imax
    else:
        return foo(S)

def tau(state, sys):
    S, E, Is, Ia, R = state
    foo = scipy.interpolate.interp1d(sys.tau.s, sys.tau.i, kind = "cubic")
    if S < sys.sbar:
        return sys.imax
    else:
        return foo(S)

def cc(state, sys):
    S, E, Is, Ia, R = state
    foo = scipy.interpolate.interp1d(sys.commutation_curve[0],
                                     sys.commutation_curve[1],
                                     kind = "cubic")
    return foo(S)

def cc_inv(state, sys):
    S, E, Is, Ia, R = state
    foo = scipy.interpolate.interp1d(sys.commutation_curve[1],
                                     sys.commutation_curve[0],
                                     kind = "cubic")
    return foo(Is)

def final_cond(t, state, u_func, ref, crit, term_val):
    out = state[0] - term_val
    return out

final_cond.terminal = True

#%%
imax = 0.1
umax = 0.6
gamma = 1/7
beta = 0.52
A = sir.SIR()
A.set_params([imax, umax, gamma, beta], flag = "bg")
sir.find_commutation_curve(A)

#%%
x0 = np.array([1 - 1e-4, 1e-4, 0, 0, 0]).reshape([5, ])
x0 = np.array([0.5, 0, 0.075, 0, 0]).reshape([5, ])
p = ref(0.52, 1/7, 1/7, 0.8)
sol = scipy.integrate.solve_ivp(SEIIR, [0, 1000], x0,
                                #method = "LSODA", min_step = 1e-3,
                                method = "RK23",
                                args = (u_opt, p, A, A.sbar),
                                events = [final_cond])
S, E, Is, Ia, R = sol.y

sol2 = scipy.integrate.solve_ivp(SEIIR, [0, 1000], x0,
                                 #method = "LSODA", min_step = 1e-3,
                                 method = "RK23",
                                 args = (u_0, p, A, A.sbar),
                                 events = [final_cond])
S2, E2, Is2, Ia2, R2 = sol2.y

sol3 = scipy.integrate.solve_ivp(SEIIR, [0, 1000], x0,
                                 #method = "LSODA", min_step = 1e-3,
                                 method = "RK23",
                                 args = (u_max, p, A, A.sbar),
                                 events = [final_cond])
S3, E3, Is3, Ia3, R3 = sol3.y

#%%
fig, ax = plt.subplots(2, 1, figsize = (18, 12))
ax[0].set_xlim(A.sbar, 1)
ax[0].set_ylim(0, A.imax*2)
ax[0].plot(A.tau.s, A.tau.i, "xkcd:blue", alpha = 0.5)
ax[0].plot(A.phi.s, A.phi.i, "xkcd:red", alpha = 0.5)
ax[0].plot(A.commutation_curve[0], A.commutation_curve[1],
           "xkcd:black", alpha = 0.5)
ax[0].plot(S, Is, "xkcd:blue")
ax[0].plot(S2, Is2, "xkcd:red", linestyle = "dotted")
ax[0].plot(S3, Is3, "xkcd:black", linestyle = "dashed")

ax[1].plot(sol.t, S, "xkcd:blue")
ax[1].plot(sol.t, E, "xkcd:pink")
ax[1].plot(sol.t, Is, "xkcd:red")
ax[1].plot(sol.t, Ia, "xkcd:green")
ax[1].plot(sol.t, R, "xkcd:black")
ax[1].plot(sol2.t, S2, "xkcd:blue", ls = "dotted")
ax[1].plot(sol2.t, E2, "xkcd:pink", ls = "dotted")
ax[1].plot(sol2.t, Is2, "xkcd:red", ls = "dotted")
ax[1].plot(sol2.t, Ia2, "xkcd:green", ls = "dotted")
ax[1].plot(sol2.t, R2, "xkcd:black", ls = "dotted")
ax[1].plot(sol3.t, S3, "xkcd:blue", ls = "dashed")
ax[1].plot(sol3.t, E3, "xkcd:pink", ls = "dashed")
ax[1].plot(sol3.t, Is3, "xkcd:red", ls = "dashed")
ax[1].plot(sol3.t, Ia3, "xkcd:green", ls = "dashed")
ax[1].plot(sol3.t, R3, "xkcd:black", ls = "dashed")
ax[1].legend(["S", "E", "Is", "Ia", "R"])

#%%
imax = 0.1
umax = 0.6
gamma = 1/7
beta = 0.52
A = sir.SIR()
A.set_params([imax, umax, gamma, beta], flag = "bg")
sir.find_commutation_curve(A)

#%%
imax = 0.1
umax = 0.6
x0 = np.array([1 - 1e-4, 0, 1e-4, 0, 0]).reshape([5, ])

rs = np.array([1.73, 2.5, 3.64])
ps = np.array([0.8, 0.7, 0.55])
ratios = np.linspace(0.7, 1.3, num = 15)

sols = np.empty([np.shape(rs)[0],
                 np.shape(ps)[0],
                 np.shape(ratios)[0]], dtype = object)
metric = np.empty(sols.shape)
dyns = np.empty(np.shape(rs), dtype = object)

for ii in range(len(rs)):
    r_dynam = rs[ii]
    beta = r_dynam * (1/7)
    print("R_0 = {}".format(r_dynam))
    Dyn = sir.SIR()
    Dyn.set_params([imax, umax, r_dynam], "r")
    print("Setting up the system for dynamics.")
    sir.find_commutation_curve(Dyn)
    dyns[ii] = Dyn
    for jj in range(len(ps)):
        p = ps[jj]
        Dyn_ref = ref(beta, 1/7, 1/7, p)
        print("\tp = {}".format(p))
        for kk in range(len(ratios)):
            print("\t\tCurrent ratio = {}".format(ratios[kk]))
            r_estim = r_dynam * ratios[kk]
            print("\t\t\\hat{{R}}_0 = {}".format(r_estim))
            Ctrol = sir.SIR()
            Ctrol.set_params([imax, umax, r_estim], "r")
            print("\t\tSetting up control system.")
            sir.find_commutation_curve(Ctrol)
            print("\t\tSimulating...")
            sol = scipy.integrate.solve_ivp(SEIIR, [0, 1e5], x0,
                                            method = "RK23", atol = 1e-7,
                                            rtol = 1e-4, events = [final_cond],
                                            args = (u_opt,
                                                    Dyn_ref,
                                                    Ctrol,
                                                    Dyn.sbar * 0.9)
                                            )
            sols[ii, jj, kk] = sol
            S, E, Is, Ia, R = sol.y
            
            m = (max(Is) - Dyn.imax) / Dyn.imax
            metric[ii, jj, kk] = m


"""
for ii in range(len(rs)):
    r_real = rs[ii]
    print("\nR_0 = {}".format(r_real))
    print("Setting up real system.")
    Dyn = sir.SIR()
    Dyn.set_params([imax, umax, r_real], "r")
    sir.find_commutation_curve(Dyn)
    dyns[ii] = Dyn
    for jj in range(len(ps)):
        p = ps[jj]
        print("\tp = {}".format(p))
        for kk in range(len(ratios)):
            r_estim = r_real * ratios[kk]
            print("\t\t\\hat{{R}}_0 = {}".format(r_estim))
            beta = r_estim * Dyn.gamma
            print("\t\t\\beta = {}".format(beta))
            Ctrol = ref(beta, 1/7, 1/7, p)
            print("\t\tSimulating...")
            sol = scipy.integrate.solve_ivp(SEIIR, [0, 1e5], x0,
                                            method = "RK23", atol = 1e-7,
                                            rtol = 1e-4, events = [final_cond],
                                            args = (u_opt, Ctrol, Dyn,
                                                    Dyn.sbar * 0.5)
                                            )
            sols[ii, jj, kk] = sol
            S, E, Is, Ia, R = sol.y
            
            m = (max(Is) - Dyn.imax) / Dyn.imax
            metric[ii, jj, kk] = m
"""



f = 2000
dur = 1
os.system("play -nq -t alsa synth {} sine {}".format(dur, f))

#%%
reference = dyns[0]
fig, ax = plt.subplots(3, 1, figsize = (18, 12))
ax[0].set_xlim(reference.sbar / 2, 1)
ax[0].set_ylim(0, reference.imax * 2)
ax[0].plot(sols[0, 0, 0].y[0], sols[0, 0, 0].y[2], "b-")
ax[0].plot(sols[0, 0, 2].y[0], sols[0, 0, 2].y[2], "r-")
ax[0].plot(sols[0, 0, 5].y[0], sols[0, 0, 5].y[2], "g-")
ax[0].plot(sols[0, 0, 8].y[0], sols[0, 0, 8].y[2], "y-")
ax[0].plot(sols[0, 0, 11].y[0], sols[0, 0, 11].y[2], "k-")
ax[0].plot(sols[0, 0, 14].y[0], sols[0, 0, 14].y[2], "c-")
ax[0].plot(reference.tau.s, reference.tau.i, "b:")
ax[0].plot(reference.phi.s, reference.phi.i, "r:")
ax[0].plot(reference.commutation_curve[0],
           reference.commutation_curve[1], "k:")
ax[0].legend(ratios[[0, 2, 5, 8, 11, 14]])
ax[0].set_title(r"$p$ = {}".format(ps[0]))

ax[1].set_xlim(reference.sbar / 2, 1)
ax[1].set_ylim(0, reference.imax * 2)
ax[1].set_ylabel(r"$I$")
ax[1].plot(sols[0, 1, 0].y[0], sols[0, 1, 0].y[2], "b-")
ax[1].plot(sols[0, 1, 2].y[0], sols[0, 1, 2].y[2], "r-")
ax[1].plot(sols[0, 1, 5].y[0], sols[0, 1, 5].y[2], "g-")
ax[1].plot(sols[0, 1, 8].y[0], sols[0, 1, 8].y[2], "y-")
ax[1].plot(sols[0, 1, 11].y[0], sols[0, 1, 11].y[2], "k-")
ax[1].plot(sols[0, 1, 14].y[0], sols[0, 1, 14].y[2], "c-")
ax[1].plot(reference.tau.s, reference.tau.i, "b:")
ax[1].plot(reference.phi.s, reference.phi.i, "r:")
ax[1].plot(reference.commutation_curve[0],
           reference.commutation_curve[1], "k:")
ax[1].legend(ratios[[0, 2, 5, 8, 11, 14]])
ax[1].set_title(r"$p$ = {}".format(ps[1]))

ax[2].set_xlim(reference.sbar / 2, 1)
ax[2].set_ylim(0, reference.imax * 2)
ax[2].set_xlabel(r"$S$")
ax[2].plot(sols[0, 2, 0].y[0], sols[0, 2, 0].y[2], "b-")
ax[2].plot(sols[0, 2, 2].y[0], sols[0, 2, 2].y[2], "r-")
ax[2].plot(sols[0, 2, 5].y[0], sols[0, 2, 5].y[2], "g-")
ax[2].plot(sols[0, 2, 8].y[0], sols[0, 2, 8].y[2], "y-")
ax[2].plot(sols[0, 2, 11].y[0], sols[0, 2, 11].y[2], "k-")
ax[2].plot(sols[0, 2, 14].y[0], sols[0, 2, 14].y[2], "c-")
ax[2].plot(reference.tau.s, reference.tau.i, "b:")
ax[2].plot(reference.phi.s, reference.phi.i, "r:")
ax[2].plot(reference.commutation_curve[0],
           reference.commutation_curve[1], "k:")
ax[2].legend(ratios[[0, 2, 5, 8, 11, 14]])
ax[2].set_title(r"$p$ = {}".format(ps[2]))

fig.suptitle(r"$R_{{0}}$ = {}".format(1.73))

fig.savefig("docs/robustness/seiir_robustness_check_173.jpg", format = "jpg")

#%%
reference = dyns[1]
fig, ax = plt.subplots(3, 1, figsize = (18, 12))
ax[0].set_xlim(reference.sbar / 2, 1)
ax[0].set_ylim(0, reference.imax * 2)
ax[0].plot(sols[1, 0, 0].y[0], sols[1, 0, 0].y[2], "b-")
ax[0].plot(sols[1, 0, 2].y[0], sols[1, 0, 2].y[2], "r-")
ax[0].plot(sols[1, 0, 5].y[0], sols[1, 0, 5].y[2], "g-")
ax[0].plot(sols[1, 0, 8].y[0], sols[1, 0, 8].y[2], "y-")
ax[0].plot(sols[1, 0, 11].y[0], sols[1, 0, 11].y[2], "k-")
ax[0].plot(sols[1, 0, 14].y[0], sols[1, 0, 14].y[2], "c-")
ax[0].plot(reference.tau.s, reference.tau.i, "b:")
ax[0].plot(reference.phi.s, reference.phi.i, "r:")
ax[0].plot(reference.commutation_curve[0],
           reference.commutation_curve[1], "k:")
ax[0].legend(ratios[[0, 2, 5, 8, 11, 14]])
ax[0].set_title(r"$p$ = {}".format(ps[0]))

ax[1].set_xlim(reference.sbar / 2, 1)
ax[1].set_ylim(0, reference.imax * 2)
ax[1].set_ylabel(r"$I$")
ax[1].plot(sols[1, 1, 0].y[0], sols[1, 1, 0].y[2], "b-")
ax[1].plot(sols[1, 1, 2].y[0], sols[1, 1, 2].y[2], "r-")
ax[1].plot(sols[1, 1, 5].y[0], sols[1, 1, 5].y[2], "g-")
ax[1].plot(sols[1, 1, 8].y[0], sols[1, 1, 8].y[2], "y-")
ax[1].plot(sols[1, 1, 11].y[0], sols[1, 1, 11].y[2], "k-")
ax[1].plot(sols[1, 1, 14].y[0], sols[1, 1, 14].y[2], "c-")
ax[1].plot(reference.tau.s, reference.tau.i, "b:")
ax[1].plot(reference.phi.s, reference.phi.i, "r:")
ax[1].plot(reference.commutation_curve[0],
           reference.commutation_curve[1], "k:")
ax[1].legend(ratios[[0, 2, 5, 8, 11, 14]])
ax[1].set_title(r"$p$ = {}".format(ps[1]))

ax[2].set_xlim(reference.sbar / 2, 1)
ax[2].set_ylim(0, reference.imax * 2)
ax[2].set_xlabel(r"$S$")
ax[2].plot(sols[1, 2, 0].y[0], sols[1, 2, 0].y[2], "b-")
ax[2].plot(sols[1, 2, 2].y[0], sols[1, 2, 2].y[2], "r-")
ax[2].plot(sols[1, 2, 5].y[0], sols[1, 2, 5].y[2], "g-")
ax[2].plot(sols[1, 2, 8].y[0], sols[1, 2, 8].y[2], "y-")
ax[2].plot(sols[1, 2, 11].y[0], sols[1, 2, 11].y[2], "k-")
ax[2].plot(sols[1, 2, 14].y[0], sols[1, 2, 14].y[2], "c-")
ax[2].plot(reference.tau.s, reference.tau.i, "b:")
ax[2].plot(reference.phi.s, reference.phi.i, "r:")
ax[2].plot(reference.commutation_curve[0],
           reference.commutation_curve[1], "k:")
ax[2].legend(ratios[[0, 2, 5, 8, 11, 14]])
ax[2].set_title(r"$p$ = {}".format(ps[2]))

fig.suptitle(r"$R_{{0}}$ = {}".format(2.5))

fig.savefig("docs/robustness/seiir_robustness_check_25.jpg", format = "jpg")

#%%
reference = dyns[2]
fig, ax = plt.subplots(3, 1, figsize = (18, 12))
ax[0].set_xlim(reference.sbar / 2, 1)
ax[0].set_ylim(0, reference.imax * 2)
ax[0].plot(sols[2, 0, 0].y[0], sols[2, 0, 0].y[2], "b-")
ax[0].plot(sols[2, 0, 2].y[0], sols[2, 0, 2].y[2], "r-")
ax[0].plot(sols[2, 0, 5].y[0], sols[2, 0, 5].y[2], "g-")
ax[0].plot(sols[2, 0, 8].y[0], sols[2, 0, 8].y[2], "y-")
ax[0].plot(sols[2, 0, 11].y[0], sols[2, 0, 11].y[2], "k-")
ax[0].plot(sols[2, 0, 14].y[0], sols[2, 0, 14].y[2], "c-")
ax[0].plot(reference.tau.s, reference.tau.i, "b:")
ax[0].plot(reference.phi.s, reference.phi.i, "r:")
ax[0].plot(reference.commutation_curve[0],
           reference.commutation_curve[1], "k:")
ax[0].legend(ratios[[0, 2, 5, 8, 11, 14]])
ax[0].set_title(r"$p$ = {}".format(ps[0]))

ax[1].set_xlim(reference.sbar / 2, 1)
ax[1].set_ylim(0, reference.imax * 2)
ax[1].set_ylabel(r"$I$")
ax[1].plot(sols[2, 1, 0].y[0], sols[2, 1, 0].y[2], "b-")
ax[1].plot(sols[2, 1, 2].y[0], sols[2, 1, 2].y[2], "r-")
ax[1].plot(sols[2, 1, 5].y[0], sols[2, 1, 5].y[2], "g-")
ax[1].plot(sols[2, 1, 8].y[0], sols[2, 1, 8].y[2], "y-")
ax[1].plot(sols[2, 1, 11].y[0], sols[2, 1, 11].y[2], "k-")
ax[1].plot(sols[2, 1, 14].y[0], sols[2, 1, 14].y[2], "c-")
ax[1].plot(reference.tau.s, reference.tau.i, "b:")
ax[1].plot(reference.phi.s, reference.phi.i, "r:")
ax[1].plot(reference.commutation_curve[0],
           reference.commutation_curve[1], "k:")
ax[1].legend(ratios[[0, 2, 5, 8, 11, 14]])
ax[1].set_title(r"$p$ = {}".format(ps[1]))

ax[2].set_xlim(reference.sbar / 2, 1)
ax[2].set_ylim(0, reference.imax * 2)
ax[2].set_xlabel(r"$S$")
ax[2].plot(sols[2, 2, 0].y[0], sols[2, 2, 0].y[2], "b-")
ax[2].plot(sols[2, 2, 2].y[0], sols[2, 2, 2].y[2], "r-")
ax[2].plot(sols[2, 2, 5].y[0], sols[2, 2, 5].y[2], "g-")
ax[2].plot(sols[2, 2, 8].y[0], sols[2, 2, 8].y[2], "y-")
ax[2].plot(sols[2, 2, 11].y[0], sols[2, 2, 11].y[2], "k-")
ax[2].plot(sols[2, 2, 14].y[0], sols[2, 2, 14].y[2], "c-")
ax[2].plot(reference.tau.s, reference.tau.i, "b:")
ax[2].plot(reference.phi.s, reference.phi.i, "r:")
ax[2].plot(reference.commutation_curve[0],
           reference.commutation_curve[1], "k:")
ax[2].legend(ratios[[0, 2, 5, 8, 11, 14]])
ax[2].set_title(r"$p$ = {}".format(ps[2]))

fig.suptitle(r"$R_{{0}}$ = {}".format(3.64))

fig.savefig("docs/robustness/seiir_robustness_check_364.jpg", format = "jpg")

#%%
fig, ax = plt.subplots(3, 1, figsize = (18, 12))
#ax[0].set_ylim(0, 1)
ax[0].plot(ratios, metric[0, 0, :], "r-")
ax[0].plot(ratios, metric[0, 1, :], "b-")
ax[0].plot(ratios, metric[0, 2, :], "k-")
ax[0].legend(ps)
#ax[1].set_ylim(0, 1)
ax[1].plot(ratios, metric[1, 0, :], "r-")
ax[1].plot(ratios, metric[1, 1, :], "b-")
ax[1].plot(ratios, metric[1, 2, :], "k-")
ax[1].legend(ps)
ax[2].plot(ratios, metric[2, 0, :], "r-")
ax[2].plot(ratios, metric[2, 1, :], "b-")
ax[2].plot(ratios, metric[2, 2, :], "k-")
ax[2].legend(ps)
ax[0].set_title(r"$R_{{0}}$ = {}".format(rs[0]))
ax[1].set_title(r"$R_{{0}}$ = {}".format(rs[1]))
ax[2].set_title(r"$R_{{0}}$ = {}".format(rs[2]))
ax[1].set_ylabel(r"$\dfrac{I_m - max(I)}{I_m}$")
ax[2].set_xlabel(r"$\dfrac{\hat{\beta}}{\beta}$")

fig.savefig("docs/robustness/seiir_robustness_rs.jpg", format = "jpg")

#%%
with open("docs/robustness/seiir_p_uncertainty_173.csv", "w") as f:
    pt = csv.writer(f, delimiter = ",")
    pt.writerow(ratios)
    pt.writerow(metric[0, 0, :])
    pt.writerow(metric[0, 1, :])
    pt.writerow(metric[0, 2, :])

with open("docs/robustness/seiir_p_uncertainty_25.csv", "w") as f:
    pt = csv.writer(f, delimiter = ",")
    pt.writerow(ratios)
    pt.writerow(metric[1, 0, :])
    pt.writerow(metric[1, 1, :])
    pt.writerow(metric[1, 2, :])

with open("docs/robustness/seiir_p_uncertainty_364.csv", "w") as f:
    pt = csv.writer(f, delimiter = ",")
    pt.writerow(ratios)
    pt.writerow(metric[2, 0, :])
    pt.writerow(metric[2, 1, :])
    pt.writerow(metric[2, 2, :])
