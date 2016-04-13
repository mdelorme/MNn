from __future__ import print_function
import scipy.optimize as op
import numpy as np
import warnings

class MNnError(Exception):
    """ 
    Miyamoto-Nagai negative exceptions : raised when the models parameters are in invalid ranges or that the user is doing something he should not
    """
    def __init__(self, msg):
        self.msg = msg
        
    def __str__(self):
        return self.msg

G = 0.0043008211
"""float: Gravitational constant to use when evaluating potential or forces on the models. 
The value must be changed to match the units required by the user."""

class MNnModel(object):
    """
    Miyamoto-Nagai negative model.
    This object is a potential-density pair expansion : it consists of a sum of Miyamoto-Nagai dics allowing 
    """
    def __init__(self, diz=1.0):
        """ Constructor for the summed Miyamoto-Nagai-negative model

        Args:
            diz (float): Normalization factor applied to all the discs (default = 1.0)
        """
        # The discs and fit description
        self.discs = []
        self.axes = []
        self.diz = diz

        # The data the model is fitting
        self.data = None
        self.yerr = None
        self.n_values = 0

    def add_disc(self, axis, a, b, M):
        """ Adds a Miyamoto-Nagai negative disc to the model, this disc will be included in the summation process when evaluating quantities with the model.

        A disc is a list of three parameters *a*, *b* and *M*. All the parameters of the discs are stored in a flat list with no real separation. 
        This is done so that emcee can be fed the array directly without any transformation.

        The model accounts for negative values of ``a``. The constraints on the parameters are the following :

          * ``b >= 0``
          * ``M >= 0``
          * ``a+b >= 0``

        Args:
            axis ({'x', 'y', 'z'}): the normal axis of the plane for the disc.
            a (float): disc scale
            b (float): disc height
            M (float): disc mass

        Raises:
            :class:`mnn.model.MNnError` : if one of the constraints if not satisfied

        Example:
            Adding a disc lying on the xy plane will be done as follows:

            >>> m = MNnModel()
            >>> m.add_disc('z', 1.0, 0.1, 10.0)
        """
        if b<0:
            raise MNnError('The height of a disc cannot be negative (b={0})'.format(b))
        elif M<0:
            raise MNnError('The mass of a disc cannot be negative (M={0})'.format(M))
        elif a+b<0:
            raise MNnError('The sum of the scale and height of a disc must be positive (a={0}, b={1})'.format(a,b))

        self.discs += [a, b, M]
        self.axes.append(axis)

    def add_discs(self, values):
        """ Wrapper for the :func:`~mnn.model.MNnModel.add_disc` method to add multiple MNn discs at the same time.
        
        Args:
            values (list of 4-tuples): The parameters of the discs to add. One 4-tuple corresponds to one disc.

        Raises:
            :class:`mnn.model.MNnError` : if one of the constraints if not satisfied

        Example:
            Adding one disc on the xy place with parameters (1.0, 0.1, 50.0) and one disc on the yz plane with parameters (1.0, 0.5, 10.0) 
            will be done as follows:

            >>> m = MNnModel()
            >>> m.add_discs([('z', 1.0, 0.1, 50.0), ('x', 1.0, 0.5, 10.0)])
        """
        for axis, a, b, M in values:
            self.add_disc(axis, a, b, M)

    def get_model(self):
        """ Copies the discs currently stored and returns them as a list of 4-tuples [(axis1, a1, b1, M1), (axis2, a2, b2, ...), ... ]
        
        Returns:
            A list of 4-tuples (axis, a, b, M).

        Example:
            >>> m = MNnModel()
            >>> m.add_discs([('z', 1.0, 0.1, 50.0), ('x', 1.0, 0.5, 10.0)])
            >>> m.get_model()
            [('z', 1.0, 0.1, 50.0), ('x', 1.0, 0.5, 10.0)]
        """
        res = []
        for id_axis, axis in enumerate(self.axes):
            res += [tuple([axis] + self.discs[id_axis*3:(id_axis+1)*3])]
        return res

    @staticmethod
    def callback_from_string(quantity):
        """ Returns the static function callback associated to a given quantity string.

        Returns:
            A function callback : One of the following : :func:`~mnn.model.MNnModel.mn_density`, :func:`~mnn.model.MNnModel.mn_potential`
        """
        cb_from_str = {'density' : MNnModel.mn_density,
                       'potential' : MNnModel.mn_potential}

        if not quantity in cb_from_str.keys():
            return MMnModel.mn_density

        return cb_from_str[quantity]

    @staticmethod
    def mn_density(r, z, a, b, M):
        """ Evaluates the density of a single Miyamoto-Nagai negative disc (a, b, M) at polar coordinates (r, z).

        Args:
            r (float): radius of the point where the density is evaluated
            z (float): height of the point where the density is evaluated
            a (float): disc scale
            b (float): disc height
            M (float): disc mass

        Returns:
            *float* : the density (scaled to the model) at (r, z)

        Note:
            This method does **not** check the validity of the constraints ``b>=0``, ``M>=0``, ``a+b>=0``
        """
        M1 = np.sqrt(M**2)
        h = np.sqrt((z**2)+(b**2))
        ah2 = (a+h)**2
        ar2 = a*(r**2)
        a3h = a+(3*h)
        num = ar2+(a3h*ah2)
        den = (h**3)*((r**2)+ah2)**2.5
        fac = (b**2)*M1/(4*np.pi)
        return fac*num/den

    @staticmethod
    def mn_potential(r, z, a, b, M):
        """ Evaluates the potential of a single Miyamoto-Nagai negative disc (a, b, M) at polar coordinates (r, z).

        Args:
            r (float): radius of the point where the density is evaluated
            z (float): height of the point where the density is evaluated
            a (float): disc scale
            b (float): disc height
            Mo (float): disc mass

        Returns:
            *float* : the potential (scaled to the model) at (r, z)

        Note:
            This method does **not** check the validity of the constraints ``b>=0``, ``M>=0``, ``a+b>=0``

        Note:
            This method relies on user-specified value for the gravitational constant. 
            This value can be overriden by setting the value :data:`mnn.model.G`.
        """
        kpc = 1000.0
        M1 = np.sqrt(M**2)
        h = np.sqrt(z**2 + b**2)
        den = r**2 + (a + h)**2
        return -G*M1 / np.sqrt(den)

    # Point evaluation
    def evaluate_potential(self, x, y, z):
        """ Evaluates the summed potential over all discs at specific positions 
        
        Args:
            x, y, z (float or Nx1 numpy array): Cartesian coordinates of the point(s) to evaluate
           
        Returns:
            The summed potential over all discs at position ``(x, y, z)``.

        Note:
            If ``x``, ``y`` and ``z`` are numpy arrays, then the return value is a Nx1 value of the potential evaluated 
            at every point ``(x[i], y[i], z[i])``
        """
        return self._evaluate_quantity(x, y, z, MNnModel.mn_potential)

    
    def evaluate_density(self, x, y, z):
        """ Evaluates the summed density over all discs at specific positions 
        
        Args:
            x, y, z (float or Nx1 numpy array): Cartesian coordinates of the point(s) to evaluate
           
        Returns:
            The summed density over all discs at position ``(x, y, z)``.

        Note:
            If ``x``, ``y`` and ``z`` are numpy arrays, then the return value is a Nx1 vector of the evaluated potential 
            at every point ``(x[i], y[i], z[i])``
        """
        return self._evaluate_quantity(x, y, z, MNnModel.mn_density)

    # Vector eval
    def evaluate_density_vec(self, x):
        """ Returns the summed density of all the discs at specific points.

        Args:
            x (Nx3 numpy array): Cartesian coordinates of the point(s) to evaluate
           
        Returns:
            The summed density over all discs at every position in vector ``x``.
        """
        return self._evaluate_quantity(x[:,0], x[:,1], x[:,2], MNnModel.mn_density)
    
    def evaluate_potential_vec(self, x):
        """ Returns the summed potential of all the discs at specific points.

        Args:
            x (Nx3 numpy array): Cartesian coordinates of the point(s) to evaluate
           
        Returns:
            The summed potential over all discs at every position in vector ``x``.
        """
        return self._evaluate_quantity(x[:,0], x[:,1], x[:,2], MNnModel.mn_potential)

    def is_positive_definite(self):
        """ Returns true if the sum of the discs are positive definite.
        
        The methods tests along every axis if the minimum of density is positive. If it is not the case then the model should 
        NOT be used since we cannot ensure positive density everywhere.

        Returns:
            A boolean indicating if the model is positive definite.
        """
        mods = self.get_model()
        
        for axis in ['x', 'y', 'z']:
            # Determine the interval
            max_range = 0.0
            for m in mods:
                # Relevant value : scale parameter for the parallel axes
                if m[0] != axis:
                    if m[1] > max_range:
                        max_range = m[1]

            # If we don't have a max_range then we can skip this root finding : the function cannot go below zero
            if abs(max_range) < 1e-18:
                continue

            max_range *= 10.0 # Multiply by a factor to be certain "everything is enclosed"

            xopt, fval, ierr, nf = op.fminbound(self._evaluate_density_axis, 0.0, max_range, args = [axis], disp=0, full_output=True)
            if fval < 0.0:
                #print('Warning : This model has a root along the {0} axis (r={1}) : density can go below zero'.format(axis, x0))
                return False

        return True

    def generate_dataset_meshgrid(self, xmin, xmax, dx, quantity='density'):
        """ Generates a numpy meshgrid of data from the model
        
        Args:
            xmin (3-tuple of floats): The low bound of the box
            xmax (3-tuple of floats): The high bound of the box
            dx (3-tuple of floats): Mesh spacing in every direction
            quantity ({'density', 'potential'}) : Type of quantity to fill the box with (default='density')

        Returns:
            A 4-tuple containing

            - **vx, vy, vz** (*N vector of floats*): The x, y and z coordinates of each point of the mesh
            - **res** (*N vector of floats*): The values of the summed quantity over all discs at each point of the mesh
        Raises:
            MemoryError: If the array is too big
            :class:`mnn.model.MNnError`: If the quantity parameter does not correspond to anything known
        """
        quantity_vec = ('density', 'potential')
        if quantity not in quantity_vec:
            print('Error : Unknown quantity type {0}, possible values are {1}'.format(quantity, quantity_vec))
            return

        if len(xmin) != 3 or len(xmax) != 3 or len(dx) != 3:
            print('Error : You must provide xmin, xmax and dx as triplets of floats')
            return

        Xsp = []
        for i in range(3):
            Xsp.append(np.linspace(xmin[i], xmax[i], (xmax[i] - xmin[i]) / dx[i] + 1))

        gx, gy, gz = np.meshgrid(Xsp[0], Xsp[1], Xsp[2])

        if quantity == 'density':
            res = self.evaluate_density(gx, gy, gz)
        elif quantity == 'potential':
            res = self.evaluate_potential(gx, gy, gz)
        else:
            raise MNnError('Quantity {0} unknown. Cannot fill grid mesh.'.format(quantity))
            
        return gx, gy, gz, res

    
    # Axis evaluation, non-documented. Should not be used apart from the is_positive_definite method ! 
    def _evaluate_density_axis(self, r, axis):
        if axis == 'x':
            return self._evaluate_quantity(r, 0, 0, MNnModel.mn_density)
        if axis == 'y':
            return self._evaluate_quantity(0, r, 0, MNnModel.mn_density)
        else:
            return self._evaluate_quantity(0, 0, r, MNnModel.mn_density)

    def _evaluate_quantity(self, x, y, z, quantity_callback):
        """ Generic private function to evaluate a quantity on the summed discs at a specific point of space.
        this function is private and should be only used indirectly via one of the following 
        :func:`~mnn.model.MNnModel.evaluate_density`, :func:`~mnn.model.MNnModel.evaluate_potential`, 
        :func:`~mnn.model.MNnModel.evaluate_density_vec`, :func:`~mnn.model.MNnModel.evaluate_potential_vec`, 

        Args:
            x, y, z (floats or Nx1 numpy arrays): Cartesian coordinates of the point(s) to evaluate
            quantity_callback (function callback): a callback indicating which function is used to evaluate the quantity

        Returns:
            *float* or *Nx1 numpy array* : The quantities evaluated at each points given in entry

        Note:
            If ``x``, ``y`` and ``z`` are numpy arrays, then the method evaluates the quantity over every point (x[i], y[i], z[i])
        """
        # Radius on each plane
        rxy = np.sqrt(x**2+y**2)
        rxz = np.sqrt(x**2+z**2)
        ryz = np.sqrt(y**2+z**2)

        # Storing the first value directly as the output variable.
        # This allows us to avoid testing for scalar or vector
        # while initializing the total_sum variable
        a, b, M = self.discs[0:3]
        axis = self.axes[0]
        if axis == "x":
            total_sum = quantity_callback(ryz, x, a, b, M)
        elif axis == "y":
            total_sum = quantity_callback(rxz, y, a, b, M)
        else:
            total_sum = quantity_callback(rxy, z, a, b, M)

        id_mod = 1
        for axis in self.axes[1:]:
            a, b, M = self.discs[id_mod*3:(id_mod+1)*3]
            if axis == "x":
                total_sum += quantity_callback(ryz, x, a, b, M)
            elif axis == "y":
                total_sum += quantity_callback(rxz, y, a, b, M)
            else:
                total_sum += quantity_callback(rxy, z, a, b, M)
            id_mod += 1
            
        return total_sum
        
    
    

    

    
