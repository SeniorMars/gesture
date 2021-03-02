import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.axes.set_xlim3d(left=-0.5, right=0.5) 
ax.axes.set_ylim3d(bottom=-0.5, top=0.5) 
ax.axes.set_zlim3d(bottom=-0.5, top=0.5) 
with h5py.File('gestures.hdf5','r') as f:
    gesture=np.array(f['/ok/ds1/'])
    def animate(i):
        global sc
        sc._offsets3d=(gesture[i,:,0],gesture[i,:,1],gesture[i,:,2])
    sc=ax.scatter(gesture[0,:,1],gesture[0,:,1],gesture[0,:,2])
    ani = matplotlib.animation.FuncAnimation(fig,animate,frames=1018,interval=10,repeat=True)
plt.show()