# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:07:29 2020

@author: Rodrigo
"""

import sir
import numpy as np
import matplotlib.pyplot as plt

#%%
r0 = 2.5
imax = 0.1

#%%
s1, i1 = [0.76, 0.01]
s2, i2 = [0.80, 0.01]
umax = np.linspace(0.0, 1.0, endpoint = False)
ni1 = np.zeros(np.shape(umax), dtype = float)
ni2 = np.zeros(np.shape(umax), dtype = float)

for ii in range(np.shape(umax)[0]):
    u = umax[ii]
    #print(u)
    B = sir.SIR()
    B.set_params([imax, u, r0], flag = "r")
    B._find_curves()
    P1 = B.add_point(s1, i1)
    P2 = B.add_point(s2, i2)
    B.find_regions()
    #print(P1.region, P2.region)
    B.get_shortest()
    M1 = P1.least_time
    M2 = P2.least_time
    tim1 = max(M1.i)
    tim2 = max(M2.i)
    ni1[ii] = (imax - tim1) / imax
    ni2[ii] = (imax - tim2) / imax

fig, ax = plt.subplots()
ax.set_xlabel(r"$u$")
#ax.set_ylim(0, 1)
ax.set_ylabel(r"$\dfrac{I_{max} - max(I)}{I_{max}}$")
ax.plot(umax, ni1)
ax.plot(umax, ni2)
ax.legend(["({}, {})".format(s1, i1), "({}, {})".format(s2, i2)])
#fig.savefig("docs/u_imax.jpg", format = "jpg")
#fig.savefig("docs/u_imax.pdf", format = "pdf")

#%%
fig, ax = plt.subplots()
ax.set_xlim(B.sbar, 1)
ax.set_ylim(0, B.imax*1.1)
ax.plot(B.tau.s, B.tau.i)
ax.plot(B.phi.s, B.phi.i)
ax.plot(B.rho.s, B.rho.i)
ax.plot(B.theta.s, B.theta.i)
ax.legend(["tau","phi","rho","theta"])
