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




## Tools





## Example 
