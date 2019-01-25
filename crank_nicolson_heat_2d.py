# Heat Equation: Parabolic PDE
# du/dt = c^2 * ddu/dx^2
# Method of solution:  Crank-Nicolson
#	The crank-nicolson method is implemented for 2-dimensions by flattening the meshgrid
#	so that the matrix operators can be applied
# Boundar Conditions Used: Constant temperature endpoints

import math
import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

def plot2D(x,y,p, title):
	from matplotlib import pyplot, cm
	from mpl_toolkits.mplot3d import Axes3D
	fig = pyplot.figure(figsize=(11, 7), dpi=100)
	ax = fig.gca(projection='3d')
	X, Y = np.meshgrid(x, y)
	surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)
	ax.set_xlim(0, int(x[-1]))
	ax.set_ylim(0, int(y[-1]))
	ax.view_init(30, 225)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	pyplot.title(title, fontsize=20)
	pyplot.show()

# In order to simplify the equations, the x and y domains are the same
nx = 50
xmin = 0
xmax = 20
ny = nx
ymin = xmin
ymax = xmax

alpha = 1.0

dx = (xmax - xmin) / (nx - 1)
dy = dx

x  = np.linspace(xmin, xmax, nx)
y = x

tmin = 0
tmax = 0.3
nt = 100

dt = (tmax-tmin)/nt
t = np.linspace(tmin, tmax, nt)

# There are no longer constraints on r, r
r = alpha * dt / dx**2

#Boundar Condition
u_boundar = 0.0

# Returns a matrix like u with the boundar condition applied
def get_bc(u, u_boundar):
	u_out = np.zeros_like(u)
	for i in range(0, u.shape[0]):
		u[i,0] = u_boundar
		u[i,-1] = u_boundar
	for j in range(0, u.shape[1]):
		u[0,j] = u_boundar
		u[-1,j] = u_boundar
	return u
	
# Flatten a square matrix u along axis 0
# First N numbers = first x row
def flatten(u):
	N = u.shape[0]
	out = np.zeros([N**2], dtype=np.float32)
	for j in range(0,N):
		out[j*N:(j+1)*N] = u[:,j]
	return out
	
def unflatten(out):
	N = int(math.sqrt(out.shape[0]))
	u = np.zeros([N,N], dtype=np.float32)
	for j in range(0, N):
		u[:,j] = out[j*N:(j+1)*N]
	return u

# Initial State
u = np.zeros([nx,ny], dtype=np.float32)
u = get_bc(u, u_boundar)
for i in range(20, 30):
	for j in range(18,32):
		u[i,j] = 1.0
plot2D(x, y, u, "Heat Equation 2-D: Initial Conditions")

print("Solving System of: "+str(nx**2)+" equations for "+str(nt)+" time steps")

# Build the A and B matricies used for solving at each time step
N = u.shape[0]-2
A = np.zeros([N**2, N**2], dtype=np.float32)	# Flattened equations
B = np.zeros([N**2, N**2], dtype=np.float32)

A[0,0] = 1+2*r		# for i=0
A[0,1] = -r/2
B[0,0] = 1-2*r
B[0,1] = r/2

for i in range(1, N):
	A[i,i] = 1+2*r
	A[i,i+1] = -r/2
	A[i,i-1] = -r/2
	
	B[i,i] = 1-2*r
	B[i,i+1] = r/2
	B[i,i-1] = r/2
	
for i in range(N, N**2 - N):
	A[i,i] = 1+2*r
	A[i,i+1] = -r/2
	A[i,i-1] = -r/2
	
	B[i,i] = 1-2*r
	B[i,i+1] = r/2
	B[i,i-1] = r/2
	
	# These matrix elements act to disperse heat in the x-direction 
	A[i,i-N] = -r/2
	A[i,i+N] = -r/2
	
	B[i,i-N] = r/2
	B[i,i+N] = r/2
	
for i in range(N**2 - N, N**2-1):
	A[i,i] = 1+2*r
	A[i,i+1] = -r/2
	A[i,i-1] = -r/2
	
	B[i,i] = 1-2*r
	B[i,i+1] = r/2
	B[i,i-1] = r/2
	
A[-1,-1] = 1+2*r
A[-1,-2] = -r/2
B[-1,-1] = 1-2*r
B[-1,-2] = r/2
# Finished building the A and B matricies

#Time-iteration loop
for ct in range(0, nt):

	b = np.zeros([u.shape[0]-2, u.shape[0]-2], dtype=np.float32)	# enforce const. temp. boundar
	b[0,:] = r*u[0,1:-1]
	b[:,0] = r*u[1:-1,0]
	b[-1,:] = r*u[-1,1:-1]
	b[:,-1] = r*u[1:-1,-1]
	b_flat = flatten(b)
	
	u_flat = flatten(u[1:-1,1:-1])
	
	BB = np.matmul(B, u_flat) + b_flat
	solved = np.linalg.solve(A,BB)
	
	u_new = get_bc(u, u_boundar)
	u_new[1:-1, 1:-1] = unflatten(solved)
	u = u_new
	
	
# Plot Solution
plot2D(x, y, u, "Crank-Nicolson Heat Equation 2-D: Final")
