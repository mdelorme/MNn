import sys

import corner
import emcee
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as op
from matplotlib.ticker import MaxNLocator

from model import MMNModel

# Thanks to Steven Bethard for this nice trick, found on :
# https://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods

# Allows the methods of MMNFitter to be pickled for multiprocessing
import copy_reg
import types

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
            return func.__get__(obj, cls)
        except KeyError:
            pass
    return None

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


sampler = None


class MMNFitter(object):
    """
    This class is used to fit a certain Multi Miyamoto Nagai model (with a predefined number of disks) to a datafile.
    """
    def __init__(self, n_walkers=100, n_steps=1000, n_threads=1, random_seed=120, fit_type='potential', check_positive_definite=False, verbose=True):
        """
        Constructor for the MultiMiyamotoNagai fitter. The fitter is based on emcee.

        :param n_walkers: emcee parameter to indicate how many parallel walkers the MCMC will use to fit the data
        :param n_steps: How many steps should the MCMC method proceed before stopping
        :param random_seed: A seed used to generate the initial positions of the walkers
        """
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_threads = n_threads

        # The fitted models
        self.samples = ()
        self.quantiles = None
        self.models = None
        self.axes = None
        self.ndim = 0
        self.fit_type = fit_type

        # The data samples we are fitting again :
        self.data = None
        self.n_values = 0
        self.yerr = None

        # Flags
        self.verbose = verbose
        self.check_DP = check_positive_definite
        if self.check_DP and self.verbose:
            print('Warning : Checking for definite-positiveness at every walker step. This ensures that the end model will be definite positive but' +
                  'might take a very long time to compute !')

        np.random.seed(random_seed)

    def set_model_type(self, nx=0, ny=0, nz=1):
        """
        Defines the type of model we are trying to fit
        :param nx: Number of disks on the yz plane
        :param ny: Number of disks on the xz plane
        :param nz: Number of disks on the xy plane
        """
        self.ndim = (nx+ny+nz)*3
        self.axes = ['x']*nx + ['y']*ny + ['z']*nz

    def load_data(self, filename):
        """
        Loads the data that needs to be fitted. The data should be in an ascii file with four columns : X Y Z potential
        :param filename: The filename to open
        """
        self.data = np.loadtxt(filename)
        self.n_values = self.data.shape[0]
        self.yerr = 0.01*np.random.rand(self.n_values)

    def sMN(self, models=(), fit_type=None):
        """
        Sums the values of the different Miyamoto on a set of points
        The type of value summed depends on the fit we want to realize
        :param models: the list of models we are summing. If none the models of the instance are taken
        :param fit_type: the type of fit we are doing. Can be "potential", "density", "force" or None. If None then
        the default fit_type of the instance is taken
        :return: a scalar, the total sum of all the models on all points
        """
        if len(models) == 0:
            models = self.models

        if not fit_type:
            fit_type = self.fit_type

        # We pick the function to apply according to the fit model
        if fit_type == 'density':
            eval_func = MMNModel.mn_density
        elif fit_type == 'potential':
            eval_func = MMNModel.mn_potential
        else:
            eval_func = MMNModel.mn_force

        # The positions of the points
        x = self.data[:, 0]
        y = self.data[:, 1]
        z = self.data[:, 2]

        # Radius on each plane
        rxy = np.sqrt(x**2+y**2)
        rxz = np.sqrt(x**2+z**2)
        ryz = np.sqrt(y**2+z**2)

        total_sum = 0.0

        # Summing on each model
        for id_mod, axis in enumerate(self.axes):
            a, b, M = models[id_mod*3:(id_mod+1)*3]
            if axis == "x":
                value = eval_func(ryz, x, a, b, M)
            elif axis == "y":
                value = eval_func(rxz, y, a, b, M)
            else:
                value = eval_func(rxy, z, a, b, M)

            total_sum += value
        return total_sum

    def loglikelihood(self, models):
        """
        This function computes the loglikelihood of the
        :param models: the list of models
        :return: the loglikelihood of the sum of models
        """

        tmp_model = MMNModel()
        
        # Checking that a+b > 0 for every model :
        for id_mod, axis in enumerate(self.axes):
            a, b, M = models[id_mod*3:(id_mod+1)*3]
            if a+b < 0:
                return -np.inf

            # If we are checking for positive-definiteness we add the disk to the model
            if self.check_DP:
                tmp_model.add_model(axis, a, b, M)

        # Now checking for positive-definiteness:
        if self.check_DP:
            if not tmp_model.is_positive_definite():
                return -np.inf

        # Everything ok, we proceed with the likelihood :
        p = self.data[:, 3]
        model = self.sMN(models)
        inv_sigma2 = 1.0/(self.yerr**2)
        return -0.5*(np.sum((p-model)**2*inv_sigma2-np.log(inv_sigma2)))

    def maximum_likelihood(self, models):
        """
        Computation of the maximum of likelihood of the models
        """
        if self.verbose:
            print("Computing maximum of likelihood")

        # Optimizing the parameters of the models to minimize the loglikelihood
        chi2 = lambda m: -2 * self.loglikelihood(m)
        result = op.minimize(chi2, models)
        values = result["x"]

        if self.verbose:
            print("Maximum of likelihood results :")

            axis_stat = {"x": [1, "yz"], "y": [1, "xz"], "z": [1, "xy"]}
            for id_mod, axis in enumerate(self.axes):
                stat = axis_stat[axis]
                axis_name = "{0}{1}".format(stat[1], stat[0])

                print("a{0} = {1}".format(axis_name, values[id_mod*3]))
                print("b{0} = {1}".format(axis_name, values[id_mod*3+1]))
                print("M{0} = {1}".format(axis_name, values[id_mod*3+2]))

                stat[0] += 1

        # Storing the best values as current models
        self.models = values

    def fit_data(self, burnin=100, x0=None, x0_range=1e-4):
        """
        This function finds the parameters of the models using emcee
        :param burnin: the number of timesteps to keep after running emcee
        :returns: A list of all the samples truncated to give only from the burning timestep
        """

        # We initialize the positions of the walkers by adding a small random component to each parameter
        if not x0:
            self.models = np.random.rand(self.ndim)
        else:
            if x0.shape != (self.ndim,):
                print("Warning : The shape given for the initial guess ({0}) is not compatible with the models ({1})".format(
                    x0.shape, (self.ndim,)))
            self.models = x0
            
        init_pos = [self.models + x0_range*np.random.randn(self.ndim) for i in range(self.n_walkers)]

        # Running the MCMC to get the parameters
        if self.verbose:
            print("Running emcee ...")

        global sampler
        sampler = emcee.EnsembleSampler(self.n_walkers, self.ndim, self.loglikelihood, threads=self.n_threads)
        sampler.run_mcmc(init_pos, self.n_steps, rstate0=np.random.get_state())

        # Storing the last burnin results
        self.samples = sampler.chain[:, burnin:, :].reshape((-1, self.ndim))

        if self.verbose:
            print("Done.")

        # Checking for positive-definiteness
        everything_dp = True
        for sample in self.samples:
            tmp_model = MMNModel()
            for id_mod, axis in enumerate(self.axes):
                  a, b, M = sample[id_mod*3:(id_mod+1)*3]
                  tmp_model.add_model(axis, a, b, M)
                  
            if not tmp_model.is_positive_definite:
                  everything_dp = False
                  break
                  
        if not everything_dp:
            print('Warning : Some sample results are not positive definite ! You can end up with negative densities')
            print('To ensure a positive definite model, consider setting the parameter "check_positive_definite" to True in the fitter !')
                  
        return self.samples

    def plot_disk_walkers(self, id_mod):
        """
        Plotting the walkers on each parameter of a certain model
        :param id_mod: the id of the disk parameters you want to plot
        """
        axis_name = {"x": "yz", "y": "xz", "z": "xy"}[self.axes[id_mod]]
        fig, axes = pl.subplots(3, 1, sharex=True, figsize=(8, 9))
        axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
        axes[0].yaxis.set_major_locator(MaxNLocator(5))
        axes[0].axhline(self.models[id_mod*3], color="#888888", lw=2)
        axes[0].set_ylabel("$a{0}$".format(axis_name))

        axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
        axes[1].yaxis.set_major_locator(MaxNLocator(5))
        axes[1].axhline(self.models[id_mod*3+1], color="#888888", lw=2)
        axes[1].set_ylabel("$b{0}$".format(axis_name))

        axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
        axes[2].yaxis.set_major_locator(MaxNLocator(5))
        axes[2].axhline(self.models[id_mod*3+2], color="#888888", lw=2)
        axes[2].set_ylabel("$M{0}$".format(axis_name))
        fig.savefig("Time.png")

    def corner_plot(self):
        """
        Draws the corner plot of the fitted data
        """
        labels = []
        axis_stat = {"x": [1, "yz"], "y": [1, "xz"], "z": [1, "xy"]}

        for id_mod, axis in enumerate(self.axes):
            stat = axis_stat[axis]
            axis_name = "{0}{1}".format(stat[1], stat[0])
            labels += ["a{0}".format(axis_name), "b{0}".format(axis_name), "M{0}".format(axis_name)]
            stat[0] += 1

        if self.verbose:
            print("Computing corner plot ...")

        figt = corner.corner(self.samples, labels=labels, truths=self.models)
        figt.savefig("Triangle.png")

    def compute_quantiles(self, quantiles=(16, 50, 84)):
        """
        Finds the quantiles values on the whole sample kept after emcee run. The results are stored in the quantiles
        attribute of the class.
        :param quantiles: a tuple indicating what quantiles are to be computed
        """
        if len(quantiles) != 3:
            sys.stderr.write('Warning : The quantile list should always be a triplet')
            return

        if len(self.samples) == 0:
            sys.stderr.write('Warning : You should not run compute_quantiles before fit_data ! Trying to compute quantiles.')

        k = lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        qarray = np.array(np.percentile(self.samples, quantiles, axis=0))
        self.quantiles = np.array((qarray[1], qarray[2]-qarray[1], qarray[1]-qarray[0])).T

        if self.verbose:
            print("MCMC results :")
            axis_stat = {"x": [1, "yz"], "y": [1, "xz"], "z": [1, "xy"]}
            for id_mod, axis in enumerate(self.axes):
                stat = axis_stat[axis]
                axis_name = "{0}{1}".format(stat[1], stat[0])
                base_format = "{0} = {1[0]} +: {1[1]} -: {1[2]}"
                print("a"+base_format.format(axis_name, self.quantiles[id_mod*3], self.quantiles[id_mod*3]))
                print("b"+base_format.format(axis_name, self.quantiles[id_mod*3+1], self.quantiles[id_mod*3+1]))
                print("M"+base_format.format(axis_name, self.quantiles[id_mod*3+2], self.quantiles[id_mod*3+2]))
                stat[0] += 1

        return self.quantiles


