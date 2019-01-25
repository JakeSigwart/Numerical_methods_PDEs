
# Laplace Equation in 2 Dimensions. Elliptic pde
# ddp/dxdx + ddp/dydy = 0
# Method of Solution: Explicit
# 	Solving through iteration until p[i,j,t] meets the conditions.

import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

def plot2D(x,y,p, title):
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

def laplace_2d(p,y,dx,dy,tolerance):
	error = 1.0
	pn = np.empty_like(p)
	while error > tolerance:
		pn = p.copy()
		p[1:-1,1:-1] = dy*dy*(pn[2:,1:-1] + pn[0:-2,1:-1])/(2*dx*dx+2*dy*dy) + dx*dx*(pn[1:-1,2:] + pn[1:-1,0:-2])/(2*dx*dx+2*dy*dy)
		
		p[0,:] = 0
		p[-1,:] = y
		p[:,0] = p[:,1]
		p[:,-1] = p[:,-2]
		error = (np.sum(np.abs(p[:])-np.abs(pn[:]))/np.sum(np.abs(pn[:])))
	return p


nx = 31
ny = 31
c = 1.0
x_length = 2.0
y_length = 2.0
dx = x_length/(nx-1)
dy = y_length/(ny-1)
p = np.zeros([nx,ny], dtype=np.float32)

x = np.linspace(0, x_length, nx)
y = np.linspace(0, y_length, ny)

#Boundary Conditions
p[0,:] = 0
p[-1,:] = y
p[:,0] = p[:,1]
p[:,-1] = p[:,-2]

plot2D(x, y, p, "Laplace Equation 2-D: Initial")

p = laplace_2d(p, y, dx, dy, 1e-4)

plot2D(x, y, p, "Laplace Equation 2-D: Final")

