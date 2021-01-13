#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:26:12 2020

@author: rmm
"""

import sir
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.optimize

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


def u_opt(state, sys, e = 1e-8):
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


def u_0(state, sys, e = 1e-8):
    return 0


def u_max(state, sys, e = 1e-8):
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
imax = 0.1
umax = 0.6
rreal = 3.64
Re = sir.SIR()
Re.set_params([imax, umax, rreal], flag = "r")
sir.find_commutation_curve(Re)

Ab = sir.SIR()
Ab.set_params([imax, umax, rreal * 1.2], flag = "r")
sir.find_commutation_curve(Ab)

Be = sir.SIR()
Be.set_params([imax, umax, rreal * 0.5], flag = "r")
sir.find_commutation_curve(Be)

sbars = [Re.sbar, Ab.sbar, Be.sbar]

#%%
fig, ax = plt.subplots(figsize = (18, 12))
ax.set_xlim(min(sbars), 1)
ax.set_ylim(0, Re.imax * 2)
ax.plot(Re.tau.s, Re.tau.i, "b-", alpha = 0.9)
ax.plot(Ab.tau.s, Ab.tau.i, "r:", alpha = 0.7)
ax.plot(Be.tau.s, Be.tau.i, "b:", alpha = 0.7)
ax.plot(Re.phi.s, Re.phi.i, "b-", alpha = 0.9)
ax.plot(Ab.phi.s, Ab.phi.i, "r:", alpha = 0.7)
ax.plot(Be.phi.s, Be.phi.i, "b:", alpha = 0.7)
ax.plot(Re.commutation_curve[0], Re.commutation_curve[1], "b-", alpha = 0.9)
ax.plot(Ab.commutation_curve[0], Ab.commutation_curve[1], "r:", alpha = 0.7)
ax.plot(Be.commutation_curve[0], Be.commutation_curve[1], "b:", alpha = 0.7)

#%%
x0 = np.array([1 - 1e-4, 1e-4, 0]).reshape([3, ])
sol = scipy.integrate.solve_ivp(F, [0, 1000], x0,
                                args = (u_opt, Re, Be, min(sbars)),
                                method = "RK23", events = [final_cond])
S, I, R = sol.y

sol2 = scipy.integrate.solve_ivp(F, [0, 1000], x0,
                                 args = (u_0, Re, Be, min(sbars)),
                                 method = "RK23", events = [final_cond])
S2, I2, R2 = sol2.y

sol3 = scipy.integrate.solve_ivp(F, [0, 1000], x0,
                                 args = (u_max, Re, Be, min(sbars)),
                                 method = "RK23", events = [final_cond])
S3, I3, R3 = sol3.y

#%%
fig, ax = plt.subplots(figsize = (18, 12))
ax.set_xlim(min(sbars), 1)
ax.set_ylim(0, Re.imax * 2)
ax.plot(S, I, "b-")
ax.plot(S2, I2, "r-", alpha = 0.7)
ax.plot(S3, I3, "k-", alpha = 0.7)
ax.plot(Re.tau.s, Re.tau.i, "b-", alpha = 0.5)
ax.plot(Ab.tau.s, Ab.tau.i, "r:", alpha = 0.5)
ax.plot(Be.tau.s, Be.tau.i, "k:", alpha = 0.5)
ax.plot(Re.phi.s, Re.phi.i, "b-", alpha = 0.5)
ax.plot(Ab.phi.s, Ab.phi.i, "r:", alpha = 0.5)
ax.plot(Be.phi.s, Be.phi.i, "k:", alpha = 0.5)
ax.plot(Re.commutation_curve[0], Re.commutation_curve[1], "b-", alpha = 0.5)
ax.plot(Ab.commutation_curve[0], Ab.commutation_curve[1], "r:", alpha = 0.5)
ax.plot(Be.commutation_curve[0], Be.commutation_curve[1], "k:", alpha = 0.5)
ax.legend([Re.sbar ** -1, Ab.sbar ** -1, Be.sbar ** -1])

#%%
imax = 0.1
umax = 0.6
rctrol = 3.64
Ctrol = sir.SIR()
Ctrol.set_params([imax, umax, rctrol], flag = "r")
sir.find_commutation_curve(Ctrol)

#%%
ratios = np.linspace(0.7, 1.3, 15)
overshoot = np.zeros(np.shape(ratios))
sols = np.empty(np.shape(ratios), dtype = object)
mods = np.empty(np.shape(ratios), dtype = object)
for ii in range(len(ratios)):
    r = rctrol / ratios[ii]
    print(r)
    Ref = sir.SIR()
    Ref.set_params([imax, umax, r], flag = "r")
    
    print("Simulating...")
    sol = scipy.integrate.solve_ivp(F, [0, 10000], x0,
                                    args = (u_opt, Ref, Ctrol, Ctrol.sbar),
                                    method = "RK23", events = [final_cond])
    S, I, R = sol.y
    overshoot[ii] = max(I)
    sols[ii] = sol
    mods[ii] = Ref

sol = scipy.integrate.solve_ivp(F, [0, 10000], x0,
                                args = (u_0, Ctrol, Ctrol, Ctrol.sbar),
                                method = "RK23", events = [final_cond])

#%%
fig, ax = plt.subplots(figsize = (18, 12))
ax.plot(ratios, (overshoot - imax) / imax)
#fig.savefig("docs/robustness/ahora_si.jpg", format = "jpg")

fig, ax = plt.subplots(figsize = (18, 12))
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.25)
ax.plot(sols[0].y[0], sols[0].y[1])
ax.plot(sols[1].y[0], sols[1].y[1])
ax.plot(sols[2].y[0], sols[2].y[1])
ax.plot(sols[3].y[0], sols[3].y[1])
ax.plot(sols[4].y[0], sols[4].y[1])
ax.plot(sols[5].y[0], sols[5].y[1])
ax.plot(sols[6].y[0], sols[6].y[1])
ax.plot(sols[7].y[0], sols[7].y[1])
ax.plot(sols[8].y[0], sols[8].y[1])
ax.plot(sols[9].y[0], sols[9].y[1])
ax.plot(sols[10].y[0], sols[10].y[1])
ax.plot(sols[11].y[0], sols[11].y[1])
ax.plot(sols[12].y[0], sols[12].y[1])
ax.plot(sols[13].y[0], sols[13].y[1])
ax.plot(sols[14].y[0], sols[14].y[1])
