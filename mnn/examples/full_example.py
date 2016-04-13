from mnn.fitter import MNnFitter
from mnn.model import MNnModel
import numpy as np
import matplotlib.pyplot as plt

print(' -- Miyamoto Nagai negative model examples --')

print('Test : Fitting data to models :')
mn = MNnModel()
fitter = MNnFitter(fit_type='density', n_threads=1, verbose=True, n_steps=1000)

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

# Initial guess :
x0 = np.array((-0.94, 2.80, 2.40, 2.91, 3.64, 19.61, 0.23, 0.67, 4.39))
samples, prob = fitter.fit_data(burnin=300, plot_freq=50, x0=x0)

fig = fitter.plot_disc_walkers()

# Getting the quantiles
quantiles = fitter.compute_quantiles(samples)
print(quantiles)
ap1 = quantiles[0][0]
bp1 = np.abs(quantiles[0][1])
Mp1 = np.abs(quantiles[0][2])
ap2 = quantiles[0][3]
bp2 = np.abs(quantiles[0][4])
Mp2 = np.abs(quantiles[0][5])
ap3 = quantiles[0][6]
bp3 = np.abs(quantiles[0][7])
Mp3 = np.abs(quantiles[0][8])

# What is the best fit value we can get from scipy :
model = quantiles[0, :]
nm, lnlike = fitter.maximum_likelihood(model)

# Creating the model according to the best fitted values
mn.add_disc('z', nm[0], nm[1], nm[2])
mn.add_disc('z', nm[3], nm[4], nm[5])
mn.add_disc('z', nm[6], nm[7], nm[8])

# Checking that the model is positive definite :
pos_def = mn.is_positive_definite()
if not pos_def:
    print('Warning : The model defined here is not positive definite : you can end up with negative densities !')
else:
    print('The model is definite positive !')

residuals = fitter.get_residuals(nm)
print(np.linalg.norm(residuals))

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
