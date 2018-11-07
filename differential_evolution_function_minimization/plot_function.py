import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pickle import dump, load

D = 2

# def f(x, y):
	# return 17*D + x**2 - 10*np.cos(2*np.pi*x + 1)  \
	            # + y**2 - 10*np.cos(2*np.pi*y + 1)

# xs = np.linspace(-5.12, 5.12, 100)
# ys = np.linspace(-5.12, 5.12, 100)

# X, Y = np.meshgrid(xs, ys)

# zs = np.array([f(x, y) for x,y in zip(np.ravel(X), np.ravel(Y))])
# Z = zs.reshape(X.shape)

# with open('points.pkl', 'wb') as file:
	# dump((X,Y,Z), file)
	
with open('points.pkl', 'rb') as file:
	X, Y, Z = load(file)

xs = X.flatten()
ys = Y.flatten()
zs = Z.flatten()

i_z = np.argmin(zs)
i_x = i_z // int(np.sqrt(len(xs)))
i_y = i_z

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(f"f({xs[i_x]:.5f}, {ys[i_y]:.5f}) = {zs[i_z]:.5f}")


R = 45
ax.plot_surface(X[R:-R], Y[R:-R], Z[R:-R], 
				  cmap=cm.coolwarm)

with open('gen_1000.txt') as file:
	lines = file.readlines()


xm = []
ym = []
zm = []
for line in lines:
	x, y, z = line.split('\t')
	x, y, z = float(x), float(y), float(z)
	
	xm.append(x)
	ym.append(y)
	zm.append(z)
	
i = zm.index(min(zm))
	
ax.plot(xm, ym, zm, color='black', marker='o', linestyle='none', markersize=3)
ax.plot([xm[i]], [ym[i]], [zm[i]], 'rX', markersize=10)
		
plt.show()
