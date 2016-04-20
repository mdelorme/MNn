import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mnn.model import MNnModel

# Single disc
model = MNnModel()
model.add_disc('z', 1.0, 10.0, 100.0)

# Evaluating density and potential :
print(model.evaluate_density(1.0, 2.0, -0.5))
print(model.evaluate_potential(1.0, 2.0, -0.5))
print(model.evaluate_force(1.0, 2.0, -0.5))

# Using vectors to evaluate density along an axis :
x = np.linspace(0.0, 30.0, 100.0)
density = model.evaluate_density(x, 0.0, 0.0)
fig = plt.plot(x, density)
plt.show()

# Plotting density meshgrid
x, y, z, v = model.generate_dataset_meshgrid((0.0, 0.0, -10.0), (30.0, 0.0, 10.0), (300, 1, 200))
fig = plt.imshow(v[:, 0].T)
plt.show()

# Contour plot
x = np.linspace(0.0, 30.0, 300)
z = np.linspace(-10.0, 10.0, 200)
plt.contour(x, z, v[:, 0].T)
plt.show()

# Plotting force meshgrid
plt.close('all')
x, y, z, f = model.generate_dataset_meshgrid((-30.0, -30.0, 0.0), (30.0, 30.0, 0.0), (30, 30, 1), 'force')
x = x[:, :, 0].reshape(-1)
y = y[:, :, 0].reshape(-1)
fx = f[0, :, :, 0].reshape(-1)
fy = f[1, :, :, 0].reshape(-1)

extent = [x.min(), x.max(), y.min(), y.max()]
plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(gs[1, 0])
pl1 = ax1.imshow(f[1, :, :, 0].T, extent=extent, aspect='auto')
ax2 = plt.subplot(gs[0, 1])
pl2 = ax2.imshow(f[0, :, :, 0].T, extent=extent, aspect='auto')
ax3 = plt.subplot(gs[1, 1])
pl3 = ax3.quiver(x.T, y.T, fx.T, fy.T, units='width', scale=0.045)
plt.show()

    
