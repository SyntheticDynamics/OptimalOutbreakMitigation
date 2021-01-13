#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:51:55 2020

@author: rmm
"""

import sir
import matplotlib.pyplot as plt

A = sir.SIR()
A.set_params([0.1, 0.5, 2], flag = "r")
sir.find_commutation_curve(A)

IC = sir.create_initial_conditions(A, 1e-4, 1e-4, end = 1.514752030974275)

#C = sir.CurveSegment(A.sbar, 0.1, 0, A, )

P = A.add_point(1.3, 1e-4)
A.find_regions()
A.get_shortest()

T = P.least_time

#%%
fig, ax = plt.subplots(figsize = (18, 12))
ax.set_xlim(A.sbar, 2)
ax.set_ylim(0, A.imax*1.1)
ax.plot(A.tau.s, A.tau.i, "b-", alpha = 0.5)
ax.plot(A.phi.s, A.phi.i, "r-", alpha = 0.5)
ax.plot(A.theta.s, A.theta.i, "g-", alpha = 0.5)
ax.plot(IC[0], IC[1], "xkcd:vomit green", linestyle = "-.")
ax.plot(A.commutation_curve[0], A.commutation_curve[1], "k-", alpha = 0.5)
ax.plot(T.s, T.i, "k-")
