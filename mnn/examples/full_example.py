from mnn.fitter import MNnFitter
from mnn.model import MNnModel
import numpy as np
import matplotlib.pyplot as plt

print(' -- Miyamoto Nagai negative model examples --')

print('Test : Fitting data to models :')
mn = MNnModel()
fitter = MNnFitter(fit_type='density', n_threads=1)

# Loading the data we want to fit against
fitter.load_data('density.dat')

x = fitter.data[:, 0]
y = fitter.data[:, 1]
z = fitter.data[:, 2]

mask = (z == 0)
v1 = np.zeros((20, 20))
v2 = np.zeros((20, 20))

for p in fitter.data[:]:
    x, y, z, v = p
    if z == 0:
        nx = int(x*2)
        ny = int(y*2)
        v1[nx, ny] = v

plt.pcolor(v1)
plt.show()


# Fitting the data
fitter.set_model_type(0, 0, 3)

samples = fitter.fit_data(burnin=0)

# Getting the quantiles
quantiles = fitter.compute_quantiles()
ap1 = quantiles[0][0]
bp1 = np.abs(quantiles[1][0])
Mp1 = np.abs(quantiles[2][0])
ap2 = quantiles[3][0]
bp2 = np.abs(quantiles[4][0])
Mp2 = np.abs(quantiles[5][0])
ap3 = quantiles[6][0]
bp3 = np.abs(quantiles[7][0])
Mp3 = np.abs(quantiles[8][0])

# What is the new Chisq :
model = quantiles[:,0]
fitter.maximum_likelihood(model)

# Creating the model according to the fitted values
mn.add_model('z', ap1, bp1, Mp1)
mn.add_model('z', ap2, bp2, Mp2)
mn.add_model('z', ap3, bp3, Mp3)

# Checking that the model is positive definite :
pos_def = mn.is_positive_definite()
if not pos_def:
    print('Warning : The model defined here is not positive definite : you can end up with negative densities !')
else:
    print('The model is definite positive !')

residuals = fitter.get_residuals(model)

for i, p in enumerate(fitter.data[:]):
    x, y, z, v = p
    if z == 0:
        nx = int(x*2)
        ny = int(y*2)
        v1[nx, ny] = residuals[i]
        v2[nx, ny] = mn.evaluate_density(x, y, z)

plt.figure(1)
plt.subplot(211)
plt.pcolor(v1)
plt.subplot(212)
plt.pcolor(v2)
plt.show()
