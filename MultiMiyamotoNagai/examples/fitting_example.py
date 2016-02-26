from MultiMiyamotoNagai.fitter import MMNFitter

if __name__ == '__main__':
    print(' -- Multi Miyamoto Nagai model examples --')

    print('Test : Fitting data to models :')
    fitter = MMNFitter(n_walkers=100, n_steps=1000, fit_type='density')

    # Loading the data we want to fit against
    fitter.load_data('density.dat')

    # Fitting the data
    fitter.set_model_type(2, 2, 2)
    fitter.maximum_likelihood()
    samples = fitter.fit_data()

    # Plotting the sampler on every quantity (disk)
    for i in range(6):
        fitter.plot_disk_walkers(i)

    fitter.corner_plot()
    fitter.compute_quantiles()
