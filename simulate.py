# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:34:23 2020

@author: Rodrigo
"""

import numpy as np
import scipy.optimize
import scipy.integrate
import matplotlib.pyplot as plt
import sir as SIR

imax = 0.1
gamma = 0.2
beta = 0.5
u = 0.5

def ds(state, t):
    s0 = state[0]
    i0 = state[1]
    return -(1-u)*beta*s0*i0

def di(state, t):
    s0 = state[0]
    i0 = state[1]
    return (1-u)*beta*s0*i0 - gamma*i0

def dt(state, t):
    return 1

def sir(state, t):
    """
    A
    """
    dF = np.array([ds(state, 0), di(state, 0), dt(state, 0)]).reshape([3, ])
    return dF

#%%
x0 = np.array([0.6, 0.08, 0])
x = sir(x0, 0)
print(x)

t = np.linspace(0, 13.178045800119373)
series = scipy.integrate.odeint(sir, x0, t)

imax = 0.1
umax = 0.5
gamma = 0.2
beta = 0.5
B = SIR.SIR()
B.set_params(imax, umax, gamma, beta)
B.find_tau()
T = B.tau
T.get_time(2, 1)

L = SIR.CurveSegment(0.6, 0.08, 0.5, B, 0.5)

plt.plot(series[:, 0], series[:, 1], "rx")
plt.plot(L.s, L.i, "r.")
#plt.plot(x0[0], x0[1], "bx")
#plt.xlim([0, 1])
#plt.ylim([0, x0[1]*1.1])
plt.xlabel("S")
plt.ylabel("I")
plt.legend(["Simulated Curve", "Calculated Curve"])
plt.savefig("docs/integral.pdf", format = "pdf")
