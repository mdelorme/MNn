import numpy as np
import matplotlib.pyplot as plt

from mnn.model import MNnModel

# Single disc
model = MNnModel()
model.add_disc('z', 1.0, 10.0, 100.0)

# Evaluating density and potential :
print(model.evaluate_density(1.0, 2.0, -0.5))
print(model.evaluate_potential(1.0, 2.0, -0.5))

# Using vectors to evaluate density along an axis :
x = np.linspace(0.0, 30.0, 100.0)
density = model.evaluate_density(x, 0.0, 0.0)
fig = plt.plot(x, density)
plt.show()

# Plotting density meshgrid
x, y, z, v = model.generate_dataset_meshgrid((0.0, 0.0, -10.0), (30.0, 0.0, 10.0), (0.1, 0.1, 0.1))
fig = plt.imshow(v[0].T)
plt.show()

# Contour plot
x = np.arange(0.0, 30.1, 0.1)
z = np.arange(-10.0, 10.1, 0.1)
plt.contour(x, z, v[0].T)
plt.show()

    

    
