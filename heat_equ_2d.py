# Heat Equation: Parabolic PDE
# du/dt = c^2 * ddu/dx^2
# Method of solution:  Explicit method
#Physical Equs: c^2 = thermal_conductivity / density * specific_heat    (Not Yet Used)

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
	
# def compute_gradient(u):
	# u_grad = np.gradient(u)
	# return u_grad[0], u_grad[1]

nx = 50
xmin = 0
xmax = 20
ny = 50
ymin = 0
ymax = 20

alpha = 1.0

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)

x  = np.linspace(xmin, xmax, nx)
y  = np.linspace(ymin, ymax, ny)

tmin = 0
tmax = 0.3
nt = 100
dt = (tmax-tmin)/nt

rx = alpha * dt / dx**2
ry = alpha * dt / dy**2

# Must satisfy the stability condition:  r <= 0.5
if rx > 0.5 or ry > 0.5:
	print("dt changed to ensure stability")
	dt = min( (0.4*dx**2/alpha), (0.4*dy**2/alpha) )
	rx = alpha * dt / dx**2
	ry = alpha * dt / dy**2
	nt = int((tmax-tmin)/dt)
t = np.linspace(tmin, tmax, nt)
#print("dt="+str(dt)+",  dx="+str(dx)+",  dy="+str(dy))


#Boundary Conditions
u_bc = 0.0

#Initial Conditions
u = np.zeros([nx,ny], dtype=np.float32)
u_new = np.zeros([nx,ny], dtype=np.float32)
for i in range(20, 30):
	for j in range(20,30):
		u[i,j] = 0.5

		
plot2D(x, y, u, "Heat Equation 2-D: Initial Conditions")

#Time-iteration loop
for ct in range(0, nt):
	#Iterate over x
	for i in range(1, nx-1):
		for j in range(1, ny-1):
			u_new[i,j] = u[i,j] + rx*( u[i+1,j] - 2*u[i,j] + u[i-1,j] ) + ry*( u[i,j+1] - 2*u[i,j] + u[i,j-1] )
	
	#Re-apply B-C
	
	u = u_new
	
# Plot Solution
plot2D(x, y, u, "Heat Equation 2-D: Final")




