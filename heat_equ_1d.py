# Heat Equation: Parabolic pde
# du/dt = c^2 * ddu/dx^2
# Method Used: Finite-Difference, Explicit
# Boundary Conditions: Heat freely leaves ends

import numpy as np
import matplotlib.pyplot as plt

# Convenient function to plot u
def plot_u(x,u,title):
	plt.plot(x, u)
	plt.title(title, fontsize=20)
	plt.xlabel('X', fontsize=16)
	plt.ylabel('U', fontsize=16)
	plt.show()

# In order to remain stable: dt/dx^2 <= 0.5
def calc_dt(dx):
	dt = 0.4*dx*dx
	return dt
	
nx = 50
xmin = 0
xmax = 1
tmin = 0
tmax = 0.01
x  = np.linspace(xmin, xmax, nx)
dx = (xmax - xmin) / (nx - 1)

alpha = 1.0		# Material constant: Thermal Diffusivity ( k/(density*c_p) )

dt = calc_dt(dx)	# Time step is set to ensure stability
nt = int((tmax-tmin)/dt)

r = alpha * dt / dx**2		#Must be less than or equal to 1/2. This is very restrictive
	
#Boundary Conditions: Full heat loss
u_lower_bc = 0.0
u_upper_bc = 0.0

#Initial Conditions
u = np.zeros([nx], dtype=np.float64)
u_new = np.zeros([nx], dtype=np.float64)

for n in range(20, 30):
	u[n] = 1.0

plot_u(x, u, title="Heat Equation Initial Conditions")

#Time-iteration loop
for ct in range(0, nt):
	u_new[0] = u_lower_bc
	for cx in range(1, nx-1):
		u_new[cx] = (1 - 2*r)*u[cx] + r*( u[cx+1] + u[cx-1] )
	u_new[-1] = u_upper_bc
	u = u_new
	
plot_u(x, u, title="Heat Equation Final State")
