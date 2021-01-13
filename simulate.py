# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:34:23 2020

@author: Rodrigo
"""

import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import sir as SIR

imax = 0.1
gamma = 0.1
beta = 0.3
u = 0.5

def us(state):
    s = state[0]
    i = state[1]
    return s * s * i

def u0(state, sys):
    return 0

def umax(state, sys):
    return sys.umax

def ds(state, t, u_func, sys):
    s = state[0]
    i = state[1]
    u = u_func(state, sys)
    print(u)
    return -(1-u)*beta*s*i

def di(state, t, u_func, sys):
    s = state[0]
    i = state[1]
    u = u_func(state, sys)
    #print(u)
    return (1-u)*beta*s*i - gamma*i

def sir(state, t, u_func, sys):
    """
    A
    """
    print(state)
    dF = np.array([ds(state, t, u_func, sys),
                   di(state, t, u_func, sys)]).reshape([2, ])
    return dF

#%%
x0 = np.array([0.9, 0.02])
x = sir(x0, 0, us)
x2 = sir(x0, 0, umax)
print(x)
print(x2)

tspan = np.linspace(0, 13.178045800119373)
series = scipy.integrate.odeint(sir, y0 = x0, t = tspan, args = (us, ))
series2 = scipy.integrate.odeint(sir, y0 = x0, t = tspan, args = (umax, ))

fig, ax = plt.subplots()
ax.set_xlim(0.4, 1)
ax.set_ylim(0, 0.1*1.1)
ax.plot(series[:, 0], series[:, 1], color = "xkcd:blue")
ax.plot(series2[:, 0], series2[:, 1], color = "xkcd:red")

#%%
ireal = 0.1
ureal = 0.5
greal = 0.1
breal = 0.3
Re = SIR.SIR()
Re.set_params([ireal, ureal, greal, breal], flag = "bg")
SIR.find_commutation_curve(Re)

iest = 0.1
uest = 0.5
gest = 0.2
best = 0.5
Est = SIR.SIR()
Est.set_params([iest, uest, gest, best], flag = "bg")
SIR.find_commutation_curve(Est)

fig, ax = plt.subplots()
ax.set_xlim(min([Re.sbar, Est.sbar]), 1)
ax.set_ylim(0, max([Re.imax, Est.imax])*1.1)
ax.plot(Re.tau.s, Re.tau.i, "b:")
ax.plot(Re.phi.s, Re.phi.i, "r:")
ax.plot(Re.commutation_curve[0], Re.commutation_curve[1], "k:")
ax.plot(Est.tau.s, Est.tau.i, "b-")
ax.plot(Est.phi.s, Est.phi.i, "r-")
ax.plot(Est.commutation_curve[0], Est.commutation_curve[1], "k-")

#%%
def compare(s1, s2, p):
    """
    Some shit to do some stuff.
    """
    print(s1)
    print(s2)

compare(Re, Est, [0.9, 0.01])

#%%
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
    if s <= sys.commutation_curve[0][-1]:
        print("Case 1")
        if s < sys.sbar or i < tau(s):
            return 0
        return sys.umax
    elif s > sys.commutation_curve[0][-1] and s < sys.commutation_curve[0][0]:
        print("Case 2")
        if ((i > tau(s)) and (i < cc(s))) or (i > sys.imax):
            return sys.umax
        elif i > cc(s) and i < sys.imax:
            return 0
        else:
            return 0
    else:
        print("Case 3")
        if i > sys.imax:
            return sys.umax
        elif s > sys.sstar and i > phi(s):
            return sys.umax
        return 0

fig, ax = plt.subplots()
ax.set_xlim(Re.sbar, 1)
ax.set_ylim(0, Re.imax*1.1)
ax.plot(Re.tau.s, Re.tau.i, "b-")
ax.plot(Re.phi.s, Re.phi.i, "r-")
ax.plot(Re.commutation_curve[0], Re.commutation_curve[1], "k-")
c = {0: "b", 0.5: "r"}
for i in range(1000):
    ss = np.random.uniform(Re.sbar, 1)
    ii = np.random.uniform(0, Re.imax*1.1)
    cc = u_crit([ss, ii], Re)
    print(cc)
    cc = c[cc]
    ax.scatter(ss, ii, color = cc, alpha = 0.5)
    
#%%
i = 6e-2
s = Re.commutation_curve[0][-1] - 1e-4
x0 = [s, i]
tspan = np.linspace(0, 100)
series = scipy.integrate.solve_ivp(sir, y0 = x0, t_span = tspan,
                                   args = (u_crit, Re))
series2 = scipy.integrate.odeint(sir, y0 = x0, t = tspan, args = (u0, Re))
series3 = scipy.integrate.odeint(sir, y0 = x0, t = tspan, args = (umax, Re))

fig, [ax1, ax2, ax3] = plt.subplots(3)
ax1.set_xlim(min([Re.sbar, Est.sbar]), 1)
ax1.set_ylim(0, max([Re.imax, Est.imax])*1.1)
ax1.plot(Re.tau.s, Re.tau.i, "b:", alpha = 0.5)
ax1.plot(Re.phi.s, Re.phi.i, "r:", alpha = 0.5)
ax1.plot(Re.commutation_curve[0], Re.commutation_curve[1], "k:", alpha = 0.5)
ax1.plot(series[:, 0], series[:, 1], "k-")
ax1.plot(series2[:, 0], series2[:, 1], "r-", alpha = 0.5)
ax1.plot(series3[:, 0], series3[:, 1], "b-", alpha = 0.5)
ax2.plot(series[:, 0], alpha = 0.5)
ax2.plot(series2[:, 0], alpha = 0.5)
ax2.plot(series3[:, 0], alpha = 0.5)
ax3.plot(series[:, 1], alpha = 0.5)
ax3.plot(series2[:, 1], alpha = 0.5)
ax3.plot(series3[:, 1], alpha = 0.5)
