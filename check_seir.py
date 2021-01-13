#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:37:25 2020

@author: rmm
"""

import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import sir
import os
import pickle
import csv

class ref():
    """
    AN OBJECT THAT WILL HAVE JUST A SET OF PARAMETERS AS OBJECT ATTRIBUTES FOR
    REASONS.
    """
    def __init__(self, beta, gamma, lamb):
        self.beta = beta
        self.gamma = gamma
        self.lamb = lamb



def dS(t, state, u_func, dynamics, control):
    beta = dynamics.beta
    u = u_func(state, control)
    return -(1 - u) * beta * state[0] * state[2]


def dE(t, state, u_func, dynamics, control):
    beta = dynamics.beta
    u = u_func(state, control)
    lamb = dynamics.lamb
    return (1 - u) * beta * state[0] * state[2] - lamb * state[1]


def dI(t, state, u_func, dynamics, control):
    lamb = dynamics.lamb
    gamma = dynamics.lamb
    return lamb * state[1] - gamma * state[2]


def dR(t, state, u_func, dynamics, control):
    gamma = dynamics.gamma
    return gamma * state[2]


def SEIR(t, state, u_func, dynamics, control, term_val):
    S = dS(t, state, u_func, dynamics, control)
    E = dE(t, state, u_func, dynamics, control)
    I = dI(t, state, u_func, dynamics, control)
    R = dR(t, state, u_func, dynamics, control)
    dSEIR = np.array([S,
                      E,
                      I,
                      R]).reshape([4, ])
    return dSEIR


def u_crit(state, sys):
    """
    A FUNCTION THAT WILL DETERMINE THE VALUE OF THE CONTROL DEPENDING ON THE
    POSITION IN A SYSTEM. THIS FUNCTION DOES NOT RETURN A SOFTENED VERSION OF
    THE CONTROL, BUT A CONTROL WITH ABRUPT OR SOLID FRONTIERS.
    """
    s = state[0]
    i = state[2]
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


def u_ori(state, sys, e = 1e-10):
    """
    A FUNCTION THAT RETURNS A CONTROL SUCH AS IT WAS IN THE ORIGINAL
    MANUSCRIPT OF THE PAPER, WITHOUT THE COMMUTATION CURVE.
    """
    S, E, I, R = state
    if I > phi(state, sys):
        return reg(I - phi(state, sys), sys.umax, e)
    else:
        return 0


def u_opt(state, sys, e = 1e-10):
    """
    A FUNCTION THAT RETURNS A SOFT VERSION OF THE CONTROL. IF THE STATE IS
    CLOSER THAN e TO THE Phi FRONTIER, THE CONTROL IS LINEAR INSTEAD OF
    DISCRETE.
    """
    S, E, I, R = state
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
    S, E, I, R = state
    foo = scipy.interpolate.interp1d(sys.phi.s, sys.phi.i, kind = "cubic")
    if S < sys.sstar:
        return sys.imax
    else:
        return foo(S)


def tau(state, sys):
    S, E, I, R = state
    foo = scipy.interpolate.interp1d(sys.tau.s, sys.tau.i, kind = "cubic")
    if S < sys.sbar:
        return sys.imax
    else:
        return foo(S)


def cc(state, sys):
    S, E, I, R = state
    foo = scipy.interpolate.interp1d(sys.commutation_curve[0],
                                     sys.commutation_curve[1],
                                     kind = "cubic")
    return foo(S)


def final_cond(t, state, u_func, evol, crit, term_val):
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

x0 = np.array([1 - 1e-4, 0, 1e-4, 0]).reshape([4, ])
p = ref(0.52, 1/7, 1/7)
sol = scipy.integrate.solve_ivp(SEIR, [0, 200], x0,
                                args = (u_ori, p, A, A.sbar),
                                method = "LSODA", min_step = 1e-3,
                                events = [final_cond])
S, E, I, R = sol.y
sol2 = scipy.integrate.solve_ivp(SEIR, [0, 200], x0,
                                 args = (u_0, p, A, A.sbar),
                                 method = "LSODA", min_step = 1e-3,
                                 events = [final_cond])
S2, E2, I2, R2 = sol2.y
sol3 = scipy.integrate.solve_ivp(SEIR, [0, 200], x0,
                                 args = (u_max, p, A, A.sbar),
                                 method = "LSODA", min_step = 1e-3,
                                 events = [final_cond])
S3, E3, I3, R3 = sol3.y

fig, [ax1, ax2] = plt.subplots(2, 1, figsize = (18, 12))
ax1.set_xlim(A.sbar, 1)
ax1.set_xlabel(r"$S$")
ax1.set_ylim(0, A.imax * 1.5)
ax1.set_ylabel(r"$I$")
ax1.plot(S, I, "r-")
ax1.plot(S2, I2, "b-")
ax1.plot(S3, I3, "k-")
ax1.plot(A.tau.s, A.tau.i, "b:")
ax1.plot(A.phi.s, A.phi.i, "r:")
ax1.plot(A.commutation_curve[0], A.commutation_curve[1], "k:")

ax2.plot(sol.t, S, color = "xkcd:blue")
ax2.plot(sol.t, E, color = "xkcd:pink")
ax2.plot(sol.t, I, color = "xkcd:red")
ax2.plot(sol.t, R, color = "xkcd:black")
ax2.set_xlabel(r"$t$")
ax2.legend(["Healthy", "Incubating", "Sick", "Removed"])

fig.suptitle(r"$SEIR$", fontsize = 20)
#fig.savefig("docs/seir_ex.jpg", format = "jpg")


#%%
imax = 0.1
umax = 0.6
x0 = np.array([1 - 1e-4, 1e-4, 0, 0]).reshape([4, ])

rs = np.array([1.73, 2.5, 3.64])
lambdas = [1/11, 1/7, 1/5]
ratios = np.linspace(0.7, 1.3, num = 15)

sols = np.empty([np.shape(rs)[0],
                 np.shape(lambdas)[0],
                 np.shape(ratios)[0]], dtype = object)
metric = np.empty(np.shape(sols), dtype = object)
dyns = np.empty(rs.shape, dtype = object)

for ii in range(len(rs)):
    r_dynam = rs[ii]
    beta = r_dynam * (1/7)
    print("R_0 = {}".format(r_dynam))
    for jj in range(len(lambdas)):
        lamb = lambdas[jj]
        Dyn_ref = ref(beta, 1/7, lamb)
        print("\t\\lambda = {}".format(lamb))
        Dyn = sir.SIR()
        Dyn.set_params([imax, umax, r_dynam], "r")
        print("\tSetting up the system for dynamics.")
        sir.find_commutation_curve(Dyn)
        dyns[ii] = Dyn
        for kk in range(len(ratios)):
            print("\t\tCurrent ratio = {}".format(ratios[kk]))
            r_estim = r_dynam * ratios[kk]
            print("\t\t\\hat{{R}}_0 = {}".format(r_estim))
            Ctrol = sir.SIR()
            Ctrol.set_params([imax, umax, r_estim], "r")
            print("\t\tSetting up control system.")
            sir.find_commutation_curve(Ctrol)
            print("\t\tSimulating...")
            sol = scipy.integrate.solve_ivp(SEIR, [0, 1e5], x0,
                                            method = "RK23", atol = 1e-7,
                                            rtol = 1e-4, events = [final_cond],
                                            args = (u_ori,
                                                    Dyn_ref,
                                                    Ctrol,
                                                    Dyn.sbar * 0.9)
                                            )
            sols[ii, jj, kk] = sol
            S, E, I, R = sol.y
            
            m = (max(I) - Dyn.imax) / Dyn.imax
            metric[ii, jj, kk] = m


"""
for ii in range(len(rs)):
    r_real = rs[ii]
    print("\nR_0 = {}".format(r_real))
    print("Setting up real system.")
    Dyn = sir.SIR()
    Dyn.set_params([imax, umax, r_real], flag = "r")
    sir.find_commutation_curve(Dyn)
    dyns[ii] = Dyn
    for jj in range(len(lambdas)):
        lamb = lambdas[jj]
        print("\t\\lambda = {}".format(lamb))
        for kk in range(len(ratios)):
            r_estim = r_real * ratios[kk]
            print("\t\t\hat{{R}}_0 = {}".format(r_estim))
            beta = r_estim * Dyn.gamma
            print("\t\t\\beta = {}".format(beta))
            Ctrol = ref(beta, 1/7, lamb)
            print("\t\tSimulating system...")
            sol = scipy.integrate.solve_ivp(SEIR, [0, 1e5], x0,
                                            method = "RK23", atol = 1e-7,
                                            rtol = 1e-4, events = [final_cond],
                                            args = (u_opt,
                                                    Ctrol,
                                                    Dyn,
                                                    Dyn.sbar * 0.9)
                                            )
            sols[ii, jj, kk] = sol
            S, E, I, R = sol.y
            
            m = (max(I) - Dyn.imax) / Dyn.imax
            metric[ii, jj, kk] = m
"""


f = 2000
dur = 1
os.system("play -nq -t alsa synth {} sine {}".format(dur, f))

#f = open("docs/robustness/data.pckl", "wb")
#pickle.dump([rs, lambdas, ratios, sols, metric], f)
#f.close()

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
ax[0].set_title(r"$\lambda$ = {}".format(lambdas[0]))

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
ax[1].set_title(r"$\lambda$ = {}".format(lambdas[1]))

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
ax[2].set_title(r"$\lambda$ = {}".format(lambdas[2]))

fig.suptitle(r"$R_{{0}}$ = {}".format(1.73))

fig.savefig("docs/previous/seir_robustness_check_173.jpg", format = "jpg")


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
ax[0].set_title(r"$\lambda$ = {}".format(lambdas[0]))

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
ax[1].set_title(r"$\lambda$ = {}".format(lambdas[1]))

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
ax[2].set_title(r"$\lambda$ = {}".format(lambdas[2]))

fig.suptitle(r"$R_{{0}}$ = {}".format(2.5))

fig.savefig("docs/previous/seir_robustness_check_25.jpg", format = "jpg")


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
ax[0].set_title(r"$\lambda$ = {}".format(lambdas[0]))

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
ax[1].set_title(r"$\lambda$ = {}".format(lambdas[1]))

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
ax[2].set_title(r"$\lambda$ = {}".format(lambdas[2]))

fig.suptitle(r"$R_{{0}}$ = {}".format(3.64))

fig.savefig("docs/previous/seir_robustness_check_364.jpg", format = "jpg")

#%%
fig, ax = plt.subplots(3, 1, figsize = (18, 12))
#ax[0].set_ylim(0, 1)
ax[0].plot(ratios, metric[0, 0, :], "r-")
ax[0].plot(ratios, metric[0, 1, :], "b-")
ax[0].plot(ratios, metric[0, 2, :], "k-")
ax[0].legend(lambdas)
#ax[1].set_ylim(0, 1)
ax[1].plot(ratios, metric[1, 0, :], "r-")
ax[1].plot(ratios, metric[1, 1, :], "b-")
ax[1].plot(ratios, metric[1, 2, :], "k-")
ax[1].legend(lambdas)
ax[2].plot(ratios, metric[2, 0, :], "r-")
ax[2].plot(ratios, metric[2, 1, :], "b-")
ax[2].plot(ratios, metric[2, 2, :], "k-")
ax[2].legend(lambdas)
ax[0].set_title(r"$R_{{0}}$ = {}".format(rs[0]))
ax[1].set_title(r"$R_{{0}}$ = {}".format(rs[1]))
ax[2].set_title(r"$R_{{0}}$ = {}".format(rs[2]))
ax[1].set_ylabel(r"$\dfrac{I_m - max(I)}{I_m}$")
ax[2].set_xlabel(r"$\dfrac{\hat{\beta}}{\beta}$")

fig.savefig("docs/previous/seir_robustness_rs.jpg", format = "jpg")

#%%
"""
with open("docs/robustness/seir_lambda_uncertainty_173.csv", "w") as f:
    pt = csv.writer(f, delimiter = ",")
    pt.writerow(ratios)
    pt.writerow(metric[0, 0, :])
    pt.writerow(metric[0, 1, :])
    pt.writerow(metric[0, 2, :])

with open("docs/robustness/seir_lambda_uncertainty_25.csv", "w") as f:
    pt = csv.writer(f, delimiter = ",")
    pt.writerow(ratios)
    pt.writerow(metric[1, 0, :])
    pt.writerow(metric[1, 1, :])
    pt.writerow(metric[1, 2, :])

with open("docs/robustness/seir_lambda_uncertainty_364.csv", "w") as f:
    pt = csv.writer(f, delimiter = ",")
    pt.writerow(ratios)
    pt.writerow(metric[2, 0, :])
    pt.writerow(metric[2, 1, :])
    pt.writerow(metric[2, 2, :])
    
#%%
fig, ax = plt.subplots(figsize = (18, 12))
ax.plot(ratios, metric[0, :], "b--")
ax.plot(ratios, metric[1, :], "b-")
ax.plot(ratios, metric[2, :], "b:")
ax.legend(["lambda = 1/11", "lambda = 1/7", "lambda = 1/5"])

#fig.savefig("docs/seir_robustness_joined.jpg", format = "jpg")"""
