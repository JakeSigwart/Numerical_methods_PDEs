# Wave Equation: Hyperbolic PDE
# Method of Solution: Finite difference, explicit

import math
import numpy as np
import matplotlib.pyplot as plt

nx = 50
xmin = 0
xmax = 2
nt  = 100
tmin = 0.0
tmax = 1.0

c = 1.0

dx = (xmax - xmin) / (nx - 1)
dt = (xmax - xmin) / (nx - 1)
x  = np.linspace(xmin, xmax, nx)
u = np.zeros_like(x)

r = c*dt/dx
# Check to make sure the method is stable
if r > 1.0:
	print("Warning: Method is unstable with the current parameters!")

# Initial Conditions: u(x,t=0) is a centered cosine wave
def set_u_initial(x,u):
	mag = 1.0
	xmin = min(x)
	xmax = max(x)
	num = x.shape[0]
	center = (xmax-xmin)/2
	freq = num/8
	for i in range(int(num/4), int(3*num/4)):
		u[i] = mag*math.cos( (x[i]-center)*freq ) + mag
	return u
	
# Estimate the u at the next time step 
def get_next_u(u,c,dt):
	u_next = np.zeros_like(u)
	for i in range(1, u.shape[0]-1):		
		u_next[i] = u[i] + 0.5*(u[i+1] - 2*u[i] + u[i-1])*r**2
	return u_next
	
# Convenient function to plot u
def plot_u(x,u,title):
	plt.plot(x, u)
	plt.title(title, fontsize=20)
	plt.xlabel('X', fontsize=16)
	plt.ylabel('U', fontsize=16)
	plt.show()
	
u_previous = set_u_initial(x,u)
u = get_next_u(u_previous,c,dt)

plot_u(x,u,"Wave Equation: Initial Condition")
u_initial = u_previous
u_new = np.zeros_like(u)

# Time-iteration loop
for t in range(0, nt):
	for i in range(1, nx-1):
		u_new[i] = r**2*u[i+1] + 2*(1-r**2)*u[i] + r**2*u[i-1] - u_previous[i]
	# Apply boundary conditions
	u[0] = 0
	u[-1] = 0
	u_previous = u
	u = u_new
	

plt.plot(x, u_initial, 'r', x, u, 'b')
plt.title("Wave Equation: Final", fontsize=20)
plt.xlabel('X', fontsize=16)
plt.ylabel('U', fontsize=16)
plt.legend(['Initial', 'Final'])
plt.show()


