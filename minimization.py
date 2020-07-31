# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:43:17 2020

@author: Rodrigo
"""

import sir
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np


#%%
def Tt(s_c, point, system):
    """
    A FUNCTION THAT CALCULATES THE TIME OF THE TRAJECTORY BY A GIVEN
    CONMUTATION POINT s.
    """
    Tx = tra(s_c, point, system)
    Tx.get_time()
    return Tx.time


def tra(s_c, point, system):
    """
    A FUNCTION THAT CALCULATES THE TIME OF THE TRAJECTORY BY A GIVEN
    CONMUTATION POINT s.
    """
    try:
        s_c = s_c[0]
    except:
        pass
        #print("Input is already a float.")
    #print("The given conmutation point is: {}".format(s_c))
    u0_curve = sir.CurveSegment(point.s0, point.i0, 0, system)
    sc, ic = u0_curve._curve_sol(system.imax)
    #print("The intersection point is: {}".format(sc))
    if s_c >= sc:
        #print("I'ma do it with only two thingamajigs.")
        Tu = sir.CurveSegment(point.s0, point.i0, 0, system, s_c)
        Tu.get_time()
        i_c = system._curve(s_c, point.s0, point.i0, 0)
        Tc = sir.CurveSegment(s_c, i_c, system.umax, system)
        send, iend = Tc.curve_intersection(system.tau)
        Tc = sir.CurveSegment(s_c, i_c, system.umax, system, send)
        Tc.get_time()
        #print("Tu: {}".format(Tu.time))
        #print("Tc: {}".format(Tc.time))
        #print(Tu.time + Tc.time)
        return sir.Trajectory(Tu, Tc)
    else:
        #print("I'ma have to do it with three thingamajigs.")
        Tu = sir.CurveSegment(point.s0, point.i0, 0, system, sc)
        Tu.get_time()
        Ts = sir.LineSegment(sc, s_c, system)
        Ts.get_time()
        Tc = sir.CurveSegment(s_c, system.imax, system.umax, system)
        send, iend = Tc.curve_intersection(system.tau)
        Tc = sir.CurveSegment(s_c, system.imax, system.umax, system, send)
        Tc.get_time()
        #print("Tu: {}".format(Tu.time))
        #print("Ts: {}".format(Ts.time))
        #print("Tc: {}".format(Tc.time))
        #print(Tu.time + Ts.time + Tc.time)
        return sir.Trajectory(Tu, Ts, Tc)

#%%
A = sir.SIR()
A.set_params(0.1, 0.5, 0.2, 0.5)
A.find_curves()

P = A.add_point(0.8, 1e-4)
A.find_regions()
print(P.region)
A.get_trajectories()
P.get_times()
T = P.get_least_time()

#x = Tt(1.77768879, P, A)
#print(x)

sol = scipy.optimize.minimize(Tt, x0 = P.s0,
                              args = (P, A), bounds = ((A.sbar, P.s0), ))
print(sol)

x = Tt(sol.x, P, A)
print(x)

Tm = tra(sol.x, P, A)

#%%
m = sir.MinimumTrajectory(P, A)
#t = m.make(0.8)
m.find_commutation()
print(m.commutation)
print(m.trajectory.time)
tt = m.make(m.commutation)

#%%
fig, ax = plt.subplots()
ax.set_xlim(A.sbar, 1)
ax.set_ylim(0, A.imax*1.1)
ax.plot(T.s, T.i, "b-")
#ax.plot(Tm.s, Tm.i, "r-")
ax.plot(m.trajectory.s, m.trajectory.i, "g:")
ax.plot(A.tau.s, A.tau.i)
ax.plot(A.phi.s, A.phi.i)
ax.plot(A.rho.s, A.rho.i)
ax.plot(A.theta.s, A.theta.i)
#ax.legend(["Previous", "Method"])

#%%
x0 = np.linspace(0.7, 1)

Ms = np.empty(np.shape(x0), dtype = object)
for ii in range(len(x0)):
    Px = A.add_point(x0[ii], 0.04)
    A.find_regions()
    Mx = sir.MinimumTrajectory(Px, A)
    Mx.find_commutation()
    Ms[ii] = Mx.trajectory

fig, ax = plt.subplots()
ax.set_xlim(0.4, 1)
ax.set_ylim(0, 0.11)
ax.plot(A.tau.s, A.tau.i, "b-")
ax.plot(A.phi.s, A.phi.i, "r-")
ax.plot(Ms[0].s, Ms[0].i, "b:")
ax.plot(Ms[9].s, Ms[9].i, "b:")
ax.plot(Ms[19].s, Ms[19].i, "b:")
ax.plot(Ms[29].s, Ms[29].i, "b:")
ax.plot(Ms[39].s, Ms[39].i, "b:")
ax.plot(Ms[49].s, Ms[49].i, "b:")
