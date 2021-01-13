#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:19:12 2020

@author: rmm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:37:06 2020

@author: rmm
"""

import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import sir
import os

class ref():
    """
    AN OBJECT THAT WILL HAVE JUST A SET OF PARAMETERS AS OBJECT ATTRIBUTES FOR
    REASONS.
    """
    def __init__(self, beta, gamma, p):
        self.beta = beta
        self.gamma = gamma
        self.p = p



def dS(t, state, u_func, ref, crit):
    S, Is, Ia, R = state
    beta = ref.beta
    u = u_func(state, crit)
    #print(u)
    return -(1 - u) * beta * S * (Ia + Is)

def dIs(t, state, u_func, ref, crit):
    S, Is, Ia, R = state
    beta = ref.beta
    p = ref.p
    gamma = ref.gamma
    u = u_func(state, crit)
    return p * (1 - u) * beta * S * (Ia + Is) - gamma * Is

def dIa(t, state, u_func, ref, crit):
    S, Is, Ia, R = state
    beta = ref.beta
    gamma = ref.gamma
    p = ref.p
    u = u_func(state, crit)
    return (1 - p) * (1 - u) * beta * S * (Ia + Is) - gamma * Ia

def dR(t, state, u_func, ref, crit):
    S, Is, Ia, R = state
    gamma = ref.gamma
    return gamma * (Ia + Is)

def SIIR(t, state, u_func, ref, crit, term_val):
    #print(state)
    S = dS(t, state, u_func, ref, crit)
    Is = dIs(t, state, u_func, ref, crit)
    Ia = dIa(t, state, u_func, ref, crit)
    R = dR(t, state, u_func, ref, crit)
    dSIIR = np.array([S,
                      Is,
                      Ia,
                      R]).reshape([4, ])
    return dSIIR


def u_crit(state, sys):
    s = state[0]
    i = state[1]
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


def u_crit_n(state, sys, e = 1e-6):
    s = state[0]
    i = state[1]
    dom = (sys.commutation_curve[0][-1], sys.commutation_curve[0][0])
    if i > phi(state, sys):
        if abs(i - phi(state, sys)) < e and s > dom[0]:
            return (1 / e) * abs(i - phi(state, sys)) * sys.umax
        return sys.umax
    elif i < tau(state, sys):
        return 0
    else:
        if s < dom[0]:
            if abs(i - tau(state, sys)) < e:
                return (1 / e) * abs(i - tau(state, sys)) * sys.umax
            #elif abs(i - phi(state, sys)) < e:
            #    return (1 / e) * abs(i - phi(state, sys))
            else:
                return sys.umax
        elif s > dom[1]:
            return 0
        else:
            if i > cc(state, sys):
                return 0
            else:
                if abs(i - cc(state, sys)) < e:
                    return (1 / e) * abs(i - cc(state, sys)) * sys.umax
                return sys.umax


def u_opt(state, sys, e = 1e-3):
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
    S, Is, Ia, R = state
    foo = scipy.interpolate.interp1d(sys.phi.s, sys.phi.i, kind = "cubic")
    if S < sys.sstar:
        return sys.imax
    else:
        return foo(S)

def tau(state, sys):
    S, Is, Ia, R = state
    foo = scipy.interpolate.interp1d(sys.tau.s, sys.tau.i, kind = "cubic")
    if S < sys.sbar:
        return sys.imax
    else:
        return foo(S)

def cc(state, sys):
    S, Is, Ia, R = state
    foo = scipy.interpolate.interp1d(sys.commutation_curve[0],
                                     sys.commutation_curve[1],
                                     kind = "cubic")
    return foo(S)

def cc_inv(state, sys):
    S, Is, Ia, R = state
    foo = scipy.interpolate.interp1d(sys.commutation_curve[1],
                                     sys.commutation_curve[0],
                                     kind = "cubic")
    return foo(Is)

def final_cond(t, state, u_func, ref, crit, term_val):
    out = state[0] - term_val
    return out

def final_cond2(t, state, u_func, ref, crit, term_val):
    Is = state[1]
    if Is < 1e-4:
        return 0
    return 1

final_cond.terminal = True
final_cond2.terminal = True

#%%
imax = 0.1
umax = 0.6
gamma = 1/7
beta = 0.52
A = sir.SIR()
A.set_params([imax, umax, gamma, beta], flag = "bg")
sir.find_commutation_curve(A)

#%%
x0 = np.array([1 - 1e-4, 1e-4, 0, 0]).reshape([4, ])
#x0 = np.array([0.84841638, 0.08522522, 0.02130467, 0.04505372]).reshape([4, ])
p = ref(0.52, 1/7, 0.8)
sol = scipy.integrate.solve_ivp(SIIR, [0, 1000], x0,
                                #method = "LSODA", min_step = 1e-3,
                                method = "RK23",
                                args = (u_opt, p, A, A.sbar),
                                events = [final_cond])
S, Is, Ia, R = sol.y

fig, [ax1, ax2] = plt.subplots(2, 1, figsize = (18, 12))
ax1.set_xlim(0, 1)
ax1.set_xlabel(r"$S$")
ax1.set_ylim(0, A.imax * 2)
ax1.set_ylabel(r"$I_s$")
ax1.plot(S, Is, "r-")
ax1.plot(A.tau.s, A.tau.i, "b:")
ax1.plot(A.phi.s, A.phi.i, "r:")
ax1.plot(A.commutation_curve[0], A.commutation_curve[1], "k:")

ax2.plot(sol.t, S, color = "xkcd:blue")
ax2.plot(sol.t, Is, color = "xkcd:red")
ax2.plot(sol.t, Ia, color = "xkcd:pink")
ax2.plot(sol.t, R, color = "xkcd:black")
ax2.set_xlabel(r"$t$")
ax2.legend(["Healthy", "Symptomatic", "Asymptomatic", "Removed"])

fig.suptitle(r"$SIIR$", fontsize = 20)

#%%
imax = 0.1
umax = 0.6
gamma = 1/7
beta = 0.52
A = sir.SIR()
A.set_params([imax, umax, gamma, beta], flag = "bg")
sir.find_commutation_curve(A)

#%%
x0 = np.array([1 - 1e-4, 1e-4, 0, 0]).reshape([4, ])

ps = [.8, .7, .55]
betas = np.linspace(0.5, 1.5, num = 15)

sols = np.empty([np.shape(ps)[0], np.shape(betas)[0]], dtype = object)
metric = np.empty([np.shape(ps)[0], np.shape(betas)[0]], dtype = object)

for ii in range(len(ps)):
    p = ps[ii]
    print(r"$p$ = {}".format(p))
    for jj in range(len(betas)):
        b = betas[jj] * beta
        print(r"  $\beta$ = {}".format(b))
        r = ref(b, 1/7, p)
        sol = scipy.integrate.solve_ivp(SIIR, [0, 1000], x0,
                                        method = "RK23",# min_step = 1e-3,
                                        args = (u_opt, r, A, A.sbar),
                                        events = [final_cond])
        sols[ii, jj] = sol
        S, E, I, R = sol.y
        
        m = (max(I) - A.imax) / A.imax
        metric[ii, jj] = m

f = 2000
dur = 1
os.system("play -nq -t alsa synth {} sine {}".format(dur, f))

#%%
fig, ax = plt.subplots(3, 1, figsize = (18, 12))
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, A.imax * 1.1)
ax[0].plot(sols[0, 0].y[0], sols[0, 0].y[1], "b-")
ax[0].plot(sols[0, 2].y[0], sols[0, 2].y[1], "r-")
ax[0].plot(sols[0, 5].y[0], sols[0, 5].y[1], "g-")
ax[0].plot(sols[0, 8].y[0], sols[0, 8].y[1], "y-")
ax[0].plot(sols[0, 11].y[0], sols[0, 11].y[1], "k-")
ax[0].plot(sols[0, 14].y[0], sols[0, 14].y[1], "c-")
ax[0].legend(betas[0:15:2])
ax[0].plot(A.tau.s, A.tau.i, "b:")
ax[0].plot(A.phi.s, A.phi.i, "r:")
ax[0].plot(A.commutation_curve[0], A.commutation_curve[1], "k:")
ax[0].set_title(r"$p$ = {}".format(ps[0]))

ax[1].set_xlim(0, 1)
ax[1].set_ylim(0, A.imax * 1.1)
ax[1].set_ylabel(r"$I$")
ax[1].plot(sols[1, 0].y[0], sols[1, 0].y[1], "b-")
ax[1].plot(sols[1, 2].y[0], sols[1, 2].y[1], "r-")
ax[1].plot(sols[1, 5].y[0], sols[1, 5].y[1], "g-")
ax[1].plot(sols[1, 8].y[0], sols[1, 8].y[1], "y-")
ax[1].plot(sols[1, 11].y[0], sols[1, 11].y[1], "k-")
ax[1].plot(sols[1, 14].y[0], sols[1, 14].y[1], "c-")
ax[1].legend(betas[0:15:2])
ax[1].plot(A.tau.s, A.tau.i, "b:")
ax[1].plot(A.phi.s, A.phi.i, "r:")
ax[1].plot(A.commutation_curve[0], A.commutation_curve[1], "k:")
ax[1].set_title(r"$p$ = {}".format(ps[1]))

ax[2].set_xlim(0, 1)
ax[2].set_ylim(0, A.imax * 1.1)
ax[2].set_xlabel(r"$S$")
ax[2].plot(sols[2, 0].y[0], sols[2, 0].y[1], "b-")
ax[2].plot(sols[2, 1].y[0], sols[2, 1].y[1], "r-")
ax[2].plot(sols[2, 2].y[0], sols[2, 2].y[1], "g-")
ax[2].plot(sols[2, 8].y[0], sols[2, 8].y[1], "y-")
ax[2].plot(sols[2, 11].y[0], sols[2, 11].y[1], "k-")
ax[2].plot(sols[2, 14].y[0], sols[2, 14].y[1], "c-")
ax[2].legend(betas[0:15:2])
ax[2].plot(A.tau.s, A.tau.i, "b:")
ax[2].plot(A.phi.s, A.phi.i, "r:")
ax[2].plot(A.commutation_curve[0], A.commutation_curve[1], "k:")
ax[2].set_title(r"$p$ = {}".format(ps[2]))

fig.savefig("docs/siir_robustness_check.jpg", format = "jpg")

#%%
fig, ax = plt.subplots(3, 1, figsize = (18, 12))
#ax[0].set_ylim(0, 1)
ax[0].plot(betas, metric[0, :])
#ax[1].set_ylim(0, 1)
ax[1].plot(betas, metric[1, :])
ax[2].plot(betas, metric[2, :])
ax[0].set_title(r"$p$ = {}".format(ps[0]))
ax[1].set_title(r"$p$ = {}".format(ps[1]))
ax[2].set_title(r"$p$ = {}".format(ps[2]))
ax[1].set_ylabel(r"$\dfrac{max(I) - I_m}{I_m}$")
ax[2].set_xlabel(r"$\dfrac{\hat{\beta}}{\beta}$")

fig.savefig("docs/siir_robustness.jpg", format = "jpg")

#%%
fig, ax = plt.subplots(figsize = (18, 12))
ax.plot(betas, metric[0, :], "b--")
ax.plot(betas, metric[1, :], "b-")
ax.plot(betas, metric[2, :], "b:")
ax.legend(["p = 0.8", "p = 0.7", "p = 0.55"])

fig.savefig("docs/siir_robustness_joined.jpg", format = "jpg")
