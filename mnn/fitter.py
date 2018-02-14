import sys
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from multiprocessing import Pool
from matplotlib.ticker import MaxNLocator

from .model import MNnModel, MNnError

# Thanks to Steven Bethard for this nice trick, found on :
# https://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
# Allows the methods of MNnFitter to be pickled for multiprocessing
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

class MNnFitter(object):
    """ 
    Miyamoto-Nagai negative fitter.

    This class is used to fit the parameters of a Miyamoto-Nagai negative model to data. 
    (with a predefined number of discs) to a datafile.
    """
    def __init__(self, n_walkers=100, n_steps=1000, n_threads=1, random_seed=123,
                 fit_type='density', check_positive_definite=False, cdp_range=None, 
                 allow_negative_mass=False, verbose=False):
        """ Constructor for the Miyamoto-Nagai negative fitter. The fitting is based on ``emcee``.

        Args:
            n_walkers (int): How many parallel walkers ``emcee`` will use to fit the data (default=100).
            n_step (int): The number of steps every walker should perform before stopping (default=1000).
            n_threads (int): Number of threads used to fit the data (default=1).
            random_seed (int): The random seed used for the fitting (default=123).
            fit_type ({'density', 'potential'}): What type of data is fitted (default='density').
            check_positive_definite (bool): Should the algorithm check if every walker is positive definite at every step ?
            cdp_range({float, None}): Maximum range to which check positive definiteness. If none, the criterion will be tested, for each axis on 10*max_scale_radius
            allow_negative_mass (bool): Allow the fitter to use models with negative masses (default=False)
            verbose (bool): Should the program output additional information (default=False).

        Note:
            Using ``check_positive_definite=True`` might guarantee that the density will be always positive. But
            this takes a toll on the computation. We advise to fit the data with ``check_positive_definite=False``.
            If the result is not definite positive, then switch this flag on and re-do the fitting.
        """
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_threads = n_threads

        # The fitted models
        self.samples = None
        self.discs = None
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
            print('Warning : Checking for definite-positiveness at every walker step. ' +
                  'This ensures that the end model will be definite positive but ' +
                  'might take a very long time to compute !')
        self.cdp_range = cdp_range
        self.allow_NM = allow_negative_mass

        np.random.seed(random_seed)

    def set_model_type(self, nx=0, ny=0, nz=1):
        """ Defines the type of Miyamoto-nagai negative model that will be fitted

        This method allows to set the number of discs available to put in the model along each plane.

        Args:
            nx (int): Number of discs on the yz plane (default=0).
            ny (int): Number of discs on the xz plane (default=0).
            nz (int): Number of discs on the xy plane (default=1).
        """
        self.ndim = (nx+ny+nz)*3
        self.axes = ['x']*nx + ['y']*ny + ['z']*nz

    def load_data(self, filename):
        """ Loads the data that will be fitted to the model. 

        The data should be in an ascii file with four columns tab or space separated : X Y Z quantity

        Args:
            filename (string): The filename to open.
        """
        self.data = np.loadtxt(filename)
        self.n_values = self.data.shape[0]
        self.yerr = 0.01*self.data[:,3] #np.random.rand(self.n_values)

    def loglikelihood(self, discs):
        """ Computes the log likelihood of a given model

        Args:
            discs (tuple): the list of parameters for the model stored in a flat-tuple (a1, b1, M1, a2, b2, ...)

        Returns:
            The loglikelihood of the model given in parameter
        """
        tmp_model = MNnModel()
        
        # Checking that a+b > 0 for every model :
        total_mass = 0.0
        for id_disc, axis in enumerate(self.axes):
            a, b, M = discs[id_disc*3:(id_disc+1)*3]

            # Blocking the walkers to go in "forbidden zones" : negative disc height, negative Mass, and a+b < 0
            if b <= 0:
                return -np.inf

            if M < 0 and not self.allow_NM:
                return -np.inf
            
            if a+b < 0:
                return -np.inf

            tmp_model.add_disc(axis, a, b, M)
            total_mass += M

        if total_mass < 0.0:
            return -np.inf

        # Now checking for positive-definiteness:
        if self.check_DP:
            if not tmp_model.is_positive_definite(self.cdp_range):
                return -np.inf

        # Everything ok, we proceed with the likelihood :
        p = self.data[:, 3]
        quantity_callback = MNnModel.callback_from_string(self.fit_type)
        model = tmp_model._evaluate_scalar_quantity(self.data[:, 0], self.data[:, 1], self.data[:, 2], quantity_callback)
        inv_sigma2 = 1.0/(self.yerr**2.0)
        return -0.5*(np.sum((p-model)**2.0*inv_sigma2))

    
    def maximum_likelihood(self, samples):
        """ Computation of the maximum likelihood for a given model and stores them in ``MNnFitter.model``

        Args:
            samples : The nd-array given by the fit_data routine
        Returns:
            The parameters corresponding to the maximized log likelihood
        """
        if self.verbose:
            print("Computing maximum of likelihood")

        # Optimizing the parameters of the model to minimize the loglikelihood
        best_model = -1
        best_score = -np.inf
        N = samples.shape[1]

        '''
        # TODO : Vectorize this !
        for i, s in enumerate(samples.T):
            if self.verbose and i%1000 == 999:
                sys.stdout.write('\r  - Reading chain {}/{}'.format(i+1, N))

            score = self.loglikelihood(s)
            if score > best_score:
                best_score = score
                best_model = i
        '''

        # Computing loglikelihood
        p = Pool(self.n_threads)
        scores = np.asarray(p.map(self.loglikelihood, samples.T))
        best_score = scores.max()
        best_mask = (scores == best_score)
        values = samples.T[best_mask][0]

        if self.verbose:
            sys.stdout.write('  - Reading chain {}/{}\n'.format(N, N))
            print("Maximum of likelihood results :")

            axis_stat = {"x": [1, "yz"], "y": [1, "xz"], "z": [1, "xy"]}
            for id_disc, axis in enumerate(self.axes):
                stat = axis_stat[axis]
                axis_name = "{0}{1}".format(stat[1], stat[0])

                print("a{0} = {1}".format(axis_name, values[id_disc*3]))
                print("b{0} = {1}".format(axis_name, values[id_disc*3+1]))
                print("M{0} = {1}".format(axis_name, values[id_disc*3+2]))

                stat[0] += 1

        return values, best_score

    def fit_data(self, burnin=100, x0=None, x0_range=1e-2, plot_freq=0, plot_ids=[]):
        """ Runs ``emcee`` to fit the model to the data. 

        Fills the :data:`mnn.fitter.sampler` object with the putative models and returns the burned-in data. The walkers are initialized
        randomly around position `x0` with a maximum dispersion of `x0_range`. This ball is the initial set of solutions and should be
        centered on the initial guess of what the parameters are. 

        Args:
            burnin (int): The number of timesteps to remove from every walker after the end (default=100).
            x0 (numpy array): The initial guess for the solution (default=None). If None, then x0 is determined randomly.
            x0_range (float): The radius of the inital guess walker ball. Can be either a single scalar or a tuple of size 3*n_discs (default=1e-2)
            plot_freq (int): The frequency at which the system outputs control plot (default=0). If 0, then the system does not plot anything until the end.
            plot_ids (array): The id of the discs to plot during the control plots (default=[]). If empty array, then every disc is plotted.

        Returns: 
            A tuple containing
            
            - **samples** (numpy array): A 2D numpy array holding every parameter value for every walker after timestep ``burnin``
            - **lnprobability** (numpy array): The samplers pointer to the matrix value of the log likelihood produced by each walker at every timestep after ``burnin``

        Raises:
            MNnError: If the user tries to fit the data without having called :func:`~mnn.fitter.MNnFitter.load_data` before.

        Note:
            The plots are outputted in the folder where the script is executed, in the file ``current_state.png``.
        """

        # We initialize the positions of the walkers by adding a small random component to each parameter
        if x0 == None:
            self.model = np.random.rand(self.ndim)
        else:
            if x0.shape != (self.ndim,):
                print("Warning : The shape given for the initial guess ({0}) is not compatible with the model ({1})".format(
                    x0.shape, (self.ndim,)))
            self.model = x0

        # We make sure we can treat a bulk init if necessary
        if type(x0_range) in (tuple, np.ndarray):
            x0_range = np.array(x0_range)
            
        init_pos = [self.model + self.model*x0_range*np.random.randn(self.ndim) for i in range(self.n_walkers)]

        # Running the MCMC to get the parameters
        if self.verbose:
            print("Running emcee ...")

        global sampler
        sampler = emcee.EnsembleSampler(self.n_walkers, self.ndim, self.loglikelihood, threads=self.n_threads)

        # Plot the chains regularly to see if the system has converged
        if plot_freq > 0:
            # Making sure we can plot what's asked (no more than three discs)
            if plot_ids == []:
                plot_ids = list(range(len(self.axes)))
            
            cur_step = 0
            pos = init_pos
            while cur_step < self.n_steps:
                if self.verbose:
                    sys.stdout.write('\r  . Step : {0}/{1}'.format(cur_step+1, self.n_steps))
                    sys.stdout.flush()
                pos, prob, state = sampler.run_mcmc(pos, plot_freq, rstate0=np.random.get_state())
                cur_step += plot_freq

                # Plotting the intermediate result
                fig = self.plot_disc_walkers(plot_ids)
                fig.savefig('current_state.png')
                plt.close()
            if self.verbose:           
                print('\r  . Step : {0}/{1}'.format(self.n_steps, self.n_steps))
        else:
            sampler.run_mcmc(init_pos, self.n_steps, rstate0=np.random.get_state())    


        # Storing the last burnin results
        samples = sampler.chain[:, burnin:, :].reshape((-1, self.ndim))
        lnprob = sampler.lnprobability[:, burnin:].reshape((-1))

        if self.verbose:
            print("Done.")

        # Checking for positive-definiteness
        everything_dp = True
        for sample in samples:
            tmp_model = MNnModel()
            for id_disc, axis in enumerate(self.axes):
                  a, b, M = sample[id_disc*3:(id_disc+1)*3]
                  tmp_model.add_disc(axis, a, b, M)
                  
            if not tmp_model.is_positive_definite:
                  everything_dp = False
                  break
                  
        if not everything_dp:
            warnings.warn('Some sample results are not positive definite ! You can end up with negative densities.\n' +
                          'To ensure a positive definite model, consider setting the parameter "check_positive_definite" to True in the fitter !')

        self.samples = samples
        return samples.T, lnprob.T

    def plot_disc_walkers(self, id_discs=None):
        """ Plotting the walkers on each parameter of a certain disc.

        Args:
            id_disc (int of list): the ids of the disc parameters you want to plot. If None, all the discs are plotted

        Returns:
            The matplotlib figure object. You can either plot it or save it.
        """
        # Making sure we have a list
        if not id_discs:
            id_discs = range(len(self.axes))
        elif type(id_discs) == int:
            id_discs = [id_discs]
            
        nplots = len(id_discs)
        fig, axes = plt.subplots(nplots, 3, sharex=True, figsize=(20, nplots*5))
        shape = axes.shape
        if len(shape) > 1:
            for axg in axes:
                for ax in axg:
                    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else:
            for ax in axes:
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
        
                
        for disc_id in id_discs:
            axis_name = {"x": "yz", "y": "xz", "z": "xy"}[self.axes[disc_id]]
            param_name = ['a', 'b', 'M']
            for i in range(3):
                pid = disc_id*3+i
                samples = sampler.chain[:,:,pid].T
                if nplots > 1:
                    axis = axes[disc_id][i]
                else:
                    axis = axes[i]
                        
                axis.plot(samples, color='k', alpha=10.0 / self.n_walkers)
                #axis.yaxis.set_major_locator(MaxNLocator(5))
                axis.set_ylabel('$'+param_name[i]+'_{{{0}{1}}}$'.format(axis_name, disc_id))
                axis.set_xlabel('Iteration')

        #plt.title('Parameter values for discs : ' + ', '.join(str(x) for x in id_discs))

        return fig

    def corner_plot(self, model=None):
        """ Computes the corner plot of the fitted data. 

        Note:
            If this method fails it might mean the fitting has not properly converged yet.

        Args:
            model (tuple): A flattened model or *None*. If *None* no truth value will be displayed on the plot.

        Returns:
            The corner plot object.
        """
        if self.samples == None:
            warnings.warn('corner_plot should not be called before fit_data !')
            return
        
        labels = []
        axis_stat = {"x": [1, "yz"], "y": [1, "xz"], "z": [1, "xy"]}

        for id_disc, axis in enumerate(self.axes):
            stat = axis_stat[axis]
            axis_name = "{0}{1}".format(stat[1], stat[0])
            labels += ["a{0}".format(axis_name), "b{0}".format(axis_name), "M{0}".format(axis_name)]
            stat[0] += 1

        if self.verbose:
            print("Computing corner plot ...")

        if model != None:
            figt = corner.corner(self.samples, labels=labels, truths=model)
        else:
            figt = corner.corner(self.samples, labels=labels)
            
        return figt


    def make_model(self, model):
        """ Takes a flattened model as parameter and returns a :class:`mnn.model.MNnModel` object.

        Args: 
            model (a tuple or a numpy object): The flattened model
        
        Returns: 
            A :class:`mnn.model.MNnModel` instance corresponding to the flattened model
        """
        res = MNnModel()
        for id_disc, axis in enumerate(self.axes):
            res.add_disc(axis, *model[id_disc*3:(id_disc+1)*3])

        return res

    def get_residuals(self, model):
        """ Computes the residual between the data and the model you provide as input
        
        Args:
            model (numpy array): The Ndiscs*3 parameter values of the model you want to compute the residuals on.

        Returns:
            A numpy array storing the residual value for every point of the data.

        Raises:
            MNnError: If the user tries to compute the residual without having called :func:`~mnn.fitter.MNnFitter.load_data` before.
        """
        if self.data == None:
            print('Error : No data loaded in the fitter ! You need to call "load_data" first')

        # Creating the model object from the parameters
        mmn = self.make_model(model)

        # Evaluating the residual :
        result = self.data[:,3] - mmn.evaluate_density(self.data[:,0], self.data[:,1], self.data[:,2])

        return result
        

        
