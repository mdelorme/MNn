import numpy as np

from mnn.model import MNnModel

if __name__ == '__main__':
    print(' -- Miyamoto Nagai negative3 model examples --')

    print('Test : Evaluation of a predefined model :')
    mnn = MNnModel()

    # We add three disks :
    mnn.add_model('x', 1.0, 1.0, 2.0)
    mnn.add_model('y', 2.0, 0.5, 2.0)
    mnn.add_model('z', 0.5, 0.5, 5.0)

    # We evaluate the quantities in a single point :
    print(mnn.evaluate_density(1.0, 0.0, 0.0))
    print(mnn.evaluate_potential(0.5, -1.0, 0.2))
    print(mnn.evaluate_force(0.2, 0.2, 0.2))
    print(mnn.evaluate_circular_velocity(1.5, 1.5))

    # We evaluate the density in a series of points :
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    print(mnn.evaluate_density(x, y, z))
