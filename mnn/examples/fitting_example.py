from mnn.fitter import MNnFitter
from mnn.model import MNnModel
import numpy as np

fitter = MNnFitter(n_walkers=100, n_steps=1000, fit_type='density', verbose=True)

# Loading the data we want to fit against
fitter.load_data('density.dat')

# Initial guess :
x0 = np.array((-0.94, 2.80, 2.40, 2.91, 3.64, 19.61, 0.23, 0.67, 4.39))

# Fitting the data
fitter.set_model_type(0, 0, 3)
samples, prob = fitter.fit_data(plot_freq=50, burnin=400, x0=x0)

# Plotting the sampler on every quantity (disk)
fig = fitter.plot_disc_walkers()
fig.show()
    
q = fitter.compute_quantiles(samples)
print(q)

# Building a model from the fitter
model = fitter.make_model(q[1])

# Plotting the corner plot with our selected value :
fig = fitter.corner_plot(q[1])
fig.show()
