# Heat Equation: Parabolic pde
# Method Used: Crank-Nicolson Method
#	 This method involves evaluating the implicit difference equation as a system.
# Boundary Conditions Used: Constant temperature endpoints

import numpy as np
import matplotlib.pyplot as plt

# Convenient function to plot u
def plot_u(x,u,title):
	plt.plot(x, u)
	plt.title(title, fontsize=20)
	plt.xlabel('X', fontsize=16)
	plt.ylabel('U', fontsize=16)
	plt.show()
	
xmin = 0
xmax = 1
dx = 0.02
nx = int((xmax-xmin)/dx)	
x  = np.linspace(xmin, xmax, nx)

dt = 0.001
nt = 100

alpha = 1.0		# Material constant: Thermal Diffusivity ( k/(density*c_p) )
r = alpha * dt / dx**2		# r is no longer restricted. Here, r=10

# Boundary conditions
u_lower = 0.0
u_upper = 0.0

#Initial Conditions
u = np.zeros([nx], dtype=np.float32)
u_new = np.zeros([nx], dtype=np.float32)
for n in range(int(u.shape[0]/4), int(3*u.shape[0]/4)):
	u[n] = 1.0
u[0] = u_lower
u[-1] = u_upper
	
plot_u(x, u, title="Crank-Nicolson Heat Equation: Initial Conditions")


# Construct the A and B matricies used in each step
def AB_matricies(u):
	A = np.zeros([u.shape[0]-2, u.shape[0]-2], dtype=np.float32)
	B = np.zeros([u.shape[0]-2, u.shape[0]-2], dtype=np.float32)
	
	A[0,0] = 2 + 2*r
	A[-1,-1] = 2 + 2*r
	
	B[0,0] = 2 - 2*r
	B[-1,-1] = 2 - 2*r
	
	for i in range(1, A.shape[0]-1):
		A[i,i-1] = -r
		A[i,i] = 2 + 2*r
		A[i,i+1] = -r
		
		B[i,i-1] = r
		B[i,i] = 2 - 2*r
		B[i,i+1] = r
	return A,B

A,B = AB_matricies(u)

#Time-iteration loop
for ct in range(0, nt):
	b = np.zeros([u.shape[0]-2], dtype=np.float32)	# enforce const. temp. boundary
	b[0] = r*u[0]
	b[-1] = r*u[-1]
	# print(b.shape)
	
	BB = np.matmul(B, u[1:-1]) + b
	
	u_new = np.zeros_like(u)
	u_new[1:-1] = np.linalg.solve(A,BB)		# u_new is the function at the next time step
	u_new[0] = u_lower						# Apply boundary conditions
	u_new[-1] = u_upper
	u = u_new

# Plot final result
plot_u(x, u, title="Crank-Nicolson Heat Equation: Final State")
