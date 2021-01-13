#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:37:30 2020

@author: rmm
"""

import sir
import matplotlib.pyplot as plt
import csv

imax = 0.1
r = 3.64
umax = 0.6

A = sir.SIR()
A.set_params([imax, umax, r], flag = "r")
sir.find_commutation_curve(A)

fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax * 1.1)
ax.plot(A.tau.s, A.tau.i, "b-")
ax.plot(A.phi.s, A.phi.i, "r-")
ax.plot(A.commutation_curve[0], A.commutation_curve[1], "k-")
ax.plot([A.sbar, 1], [A.imax, A.imax], "k-.")
ax.set_title("imax = {}, umax = {}, R0 = {}".format(imax, umax, r))
fig.savefig("problema_tulio3.jpg", format = "jpg")

with open("problema_tulio_3.csv", "w") as f:
    pt = csv.writer(f, delimiter = ",")
    pt.writerow(A.commutation_curve[0])
    pt.writerow(A.commutation_curve[1])
 