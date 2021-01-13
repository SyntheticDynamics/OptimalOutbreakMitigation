# SIR

## Motivation
This code was generated to solve the problem of shortest manipulations for pandemic mitigation.
In order to solve this, a SIR model
\begin{align}
	\dfrac{dS}{dt} &= - (1 - u) \beta SI \\
	\dfrac{dI}{dt} &= (1 - u) \beta SI - \gamma I \\
	\dfrac{dR}{dt} &= \gamma I
\end{align}
with a control variable $u$ is studied. To keep things simple, only two values of $u$ are considered:
$u = 0$, or no control/quarantine, and $u = u_{max}$, the maximum possible value of $u$.
This control can be considered as a non-pharmaceutical intervention, as it is a reduction in the infection
rate achieved by a reduction in citizens' mobility.

Another parameter that needs to be taken into account is the maximum hospital occupancy $I_{max}$ that can
be handled by the 'infected' community. Any 'optimal' trajectory needs to keep (or at least attempt to keep)
the infected population below this maximum occupancy.

A trajectory is considered to be in a safe area when it enters the region left of the solution
$\phi(\bar{S}, I_{max}, 0)$ that passes through $I_{max}$ and $\bar{S}$ with $u = 0$. Reaching this zone in
minimal time, then, is the goal of the paper and this code.




## Tools
The package `sir.py` included contains several classes that will aid in the identification of optimal trajectories, many
of which will not be directly visible. The most important of these is the class `SIR()`, upon which most of the work
will be done. This class contains the parameters of the system and the necessary methods to find the shortest trajectory.

The problem is divided into five regions of the plane:
1. The safe region, left of $\phi(\cdot)$.




## Example 
