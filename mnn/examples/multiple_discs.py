import numpy as np
import matplotlib.pyplot as plt

from mnn.model import MNnModel

# Creating the model
model = MNnModel()

# Putting the discs in a list for simplicity
discs = (('z', 10.0, 10.0, 100.0), ('y', -7.0, 20.0, 10.0))
model.add_discs(discs)

# Generating the meshgrid and plotting
x, y, z, v = model.generate_dataset_meshgrid((0.0, -30.0, -30.0), (0.0, 30.0, 30.0), (0.1, 0.1, 0.1))
plt.imshow(v[:,0].T)
plt.show()

# Contour plot
y = np.arange(-30.0, 30.1, 0.1)
z = np.arange(-30.0, 30.1, 0.1)
plt.contour(y, z, v[:, 0].T)
plt.show()
