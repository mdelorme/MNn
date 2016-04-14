from mnn.fitter import MNnFitter
import numpy as np

if __name__ == '__main__':
    print(' -- Miyamoto Nagai negative model examples --')

    print('Test : Fitting data to models :')
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

    fig = fitter.corner_plot()
    fig.show()
    
    q = fitter.compute_quantiles(samples)
    print(q)
