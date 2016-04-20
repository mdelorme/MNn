import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mnn.model import MNnModel

# Creating the model
model = MNnModel()

# Putting the discs in a list for simplicity
discs = (('z', 20.0, 10.0, 100.0), ('y', -12.0, 20.0, 10.0))
model.add_discs(discs)

# Generating the meshgrid and plotting
x, y, z, v = model.generate_dataset_meshgrid((0.0, -30.0, -30.0), (0.0, 30.0, 30.0), (1, 600, 600))
plt.imshow(v[0].T)
plt.show()

# Contour plot
y = np.linspace(0.0, 30.0, 600)
z = np.linspace(0.0, 30.0, 600)
plt.contour(y, z, v[0].T)
plt.show()

# Plotting force meshgrid
x, y, z, f = model.generate_dataset_meshgrid((0.0, -30.0, -30.0), (0.0, 30.0, 30.0), (1, 30, 30), 'force')
y = y[0].reshape(-1)
z = z[0].reshape(-1)
fy = f[1, 0].reshape(-1)
fz = f[2, 0].reshape(-1)

plt.close('all')
extent = [y.min(), y.max(), z.min(), z.max()]
plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(gs[1, 0])
pl1 = ax1.imshow(f[2, 0].T, extent=extent, aspect='auto')
ax2 = plt.subplot(gs[0, 1])
pl2 = ax2.imshow(f[1, 0].T, extent=extent, aspect='auto')
ax3 = plt.subplot(gs[1, 1])
pl3 = ax3.quiver(y.T, z.T, fy.T, fz.T, units='width')
plt.show()
