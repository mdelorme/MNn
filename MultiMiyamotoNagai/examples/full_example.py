from MultiMiyamotoNagai.fitter import MMNFitter
from MultiMiyamotoNagai.model import MMNModel
import numpy as np


print(' -- Multi Miyamoto Nagai model examples --')

print('Test : Fitting data to models :')
mn = MMNModel()
fitter = MMNFitter(fit_type='density')

# Loading the data we want to fit against
fitter.load_data('density.dat')

# Fitting the data
fitter.set_model_type(0, 0, 3)
fitter.maximum_likelihood()
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


