from __future__ import print_function
import scipy.optimize as op
import numpy as np

class MNnModel(object):
    """
    This object creates a model comprised of multiple Miyamoto Nagai disks fitting a given potential.
    """
    def __init__(self, diz=1.0):
        """
        Constructor for the multi miyamoto nagai disk model
        :param diz: Normalization factor applied to all the models
        """
        # The disks and fit description
        self.models = []
        self.axes = []
        self.diz = diz

        # The data the model is fitting
        self.data = None
        self.yerr = None
        self.n_values = 0

    def add_model(self, axis, a, b, M):
        """
        This function adds a disk to the model. The disk, lies on a plane normal to "axis"
        A model is a list of three values a, b, M. Those values are all stored in line with no real separation. This is
        done so that emcee can be fed the array directly without any transformation.
        :param axis: the normal axis of the plane for the disk. The axis is a string and can be "x", "y" or "z"
        :param a: disk scale
        :param b: disk height
        :param M: model amplitude
        """
        self.models += [a, b, M]
        self.axes.append(axis)

    def add_models(self, values):
        """
        This function wraps the previous method to add multiple axes at the same time
        :param values: a list of 4-tuples : (axis, a, b, M). axis is a string indicating the normal vector to the plane
        of the disk to add. Can be : "x", "y" or "z"
        """
        for axis, a, b, M in values:
            self.add_model(axis, a, b, M)

    def get_models(self):
        """
        Copies the models currently stored and returns them
        :return: a list of models.
        """
        # Grouping the models
        res = []
        for id_axis, axis in enumerate(self.axes):
            res += [[axis] + self.models[id_axis*3:(id_axis+1)*3]]
        return res

    @staticmethod
    def callback_from_string(quantity):
        """
        REturns the static function callback associated to a given quantity string 
        """
        cb_from_str = {'density' : MNnModel.mn_density,
                       'forceR' : MNnModel.mn_forceR,
                       'forceV' : MNnModel.mn_forceV,
                       'potential' : MNnModel.mn_potential}

        return cb_from_str[quantity]

    @staticmethod
    def mn_density(r, z, a, b, Mo):
        """
        Returns the Miyamoto Nagai density at polar coordinates (r, z).

        :param r: radius of the point where the density is evaluated
        :param z: height of the point where the density is evaluated
        :param a: disk scale
        :param b: disk height
        :param Mo: model amplitude
        """
        M = np.sqrt(Mo**2)
        h = np.sqrt((z**2)+(b**2))
        ah2 = (a+h)**2
        ar2 = a*(r**2)
        a3h = a+(3*h)
        num = ar2+(a3h*ah2)
        den = (h**3)*((r**2)+ah2)**2.5
        fac = (b**2)*M/(4*np.pi)
        return fac*num/den

    @staticmethod
    def mn_potential(r, z, a, b, Mo):
        """
        Returns the Miyamoto Nagai potential at polar coordinates (r, z)
        :param r: radius of the point where the density is evaluated
        :param z: height of the point where the density is evaluated
        :param a: disk scale
        :param b: disk height
        :param Mo: model amplitude
        """
        G = 0.0043008211
        kpc = 1000.0
        M = np.sqrt(Mo**2)
        h = np.sqrt(z**2 + b**2)
        den = r**2 + (a + h)**2
        return -G*M / np.sqrt(den)

    @staticmethod
    def mn_forceR(r, z, ao, bo, Mo):
        """
        Returns the radial component of Miyamoto Nagai force applied at polar coordinates (r, z)
        :param r: radius of the point where the density is evaluated
        :param z: height of the point where the density is evaluated
        :param a: disk scale
        :param b: disk height
        :param Mo: model amplitude
        """
        G = 0.0043008211
        kpc = 1000.0
        M = np.sqrt(Mo**2)
        rp=r*kpc
        zp=z*kpc
        a = ao*kpc
        b = bo*kpc
        h = np.sqrt(zp**2 + b**2)
        den = (rp**2 + (a + h)**2)**1.5
        return -G*M*rp/den

    @staticmethod
    def mn_forceV(r, z, ao, bo, Mo):
        """
        Returns the vertical component of Miyamoto Nagai force applied at polar coordinates (r, z)
        :param r: radius of the point where the density is evaluated
        :param z: height of the point where the density is evaluated
        :param a: disk scale
        :param b: disk height
        :param Mo: model amplitude
        """
        G = 0.0043008211
        kpc = 1000.0
        M = np.sqrt(Mo**2)
        rp=r*kpc
        zp=z*kpc
        a = ao*kpc
        b = bo*kpc
        h = np.sqrt(zp**2 + b**2)
        den = (rp**2 + (a + h)**2)**1.5
        num = G*M*zp
        fac = (a+h)/h
        return -fac*num/den

    @staticmethod
    def mn_circular_velocity(r, z, ao, bo, Mo):
        """
        Returns the radial component of Miyamoto Nagai force applied at polar coordinates (r, z)
        :param r: radius of the point where the density is evaluated
        :param z: height of the point where the density is evaluated
        :param a: disk scale
        :param b: disk height
        :param Mo: model amplitude
        """
        G = 0.0043008211
        kpc = 1000.0
        M = np.sqrt(Mo**2)
        rp=r*kpc
        zp=z*kpc
        a = ao*kpc
        b = bo*kpc
        h = np.sqrt(zp**2 + b**2)
        den = (rp**2 + (a + h)**2)**1.5
        return r*np.sqrt(G*M/den)

    def _evaluate_quantity(self, x, y, z, quantity_callback):
        """
        Generic private function to evaluate a quantity at a specific point of space
        :param x:
        :param y:
        :param z:
        :param quantity_callback: a function callback indicating which function is used to evaluate the quantity
        :return: the quantities evaluated at each points given in entry
        """
        # Radius on each plane
        rxy = np.sqrt(x**2+y**2)
        rxz = np.sqrt(x**2+z**2)
        ryz = np.sqrt(y**2+z**2)

        # Storing the first value firectly as the output variable.
        # This allows us to avoid testing for scalar or vector
        # while initializing the total_sum variable
        a, b, M = self.models[0:3]
        axis = self.axes[0]
        if axis == "x":
            total_sum = quantity_callback(ryz, x, a, b, M)
        elif axis == "y":
            total_sum = quantity_callback(rxz, y, a, b, M)
        else:
            total_sum = quantity_callback(rxy, z, a, b, M)

        id_mod = 1
        for axis in self.axes[1:]:
            a, b, M = self.models[id_mod*3:(id_mod+1)*3]
            if axis == "x":
                total_sum += quantity_callback(ryz, x, a, b, M)
            elif axis == "y":
                total_sum += quantity_callback(rxz, y, a, b, M)
            else:
                total_sum += quantity_callback(rxy, z, a, b, M)
            id_mod += 1
        return total_sum


    def evaluate_potential(self, x, y, z):
        """
        Returns the summed potential of all the disks at a specific point. xyz can be scalars or a vector
        """
        return self._evaluate_quantity(x, y, z, MNnModel.mn_potential)

    def evaluate_potential_vec(self, x):
        """
        Returns the summed potential of all the disks at a specific point. x is a Nx3 array
        """
        return self._evaluate_quantity(x[:,0], x[:,1], x[:,2], MNnModel.mn_potential)

    def evaluate_density(self, x, y, z):
        """
        Returns the summed density of all the disks at a specific point. xyz can be scalars or a vector
        """
        return self._evaluate_quantity(x, y, z, MNnModel.mn_density)

    def evaluate_density_vec(self, x):
        """
        Returns the summed density of all the disks at a specific point. x if a Nx3 array
        """
        return self._evaluate_quantity(x[0], x[1], x[2], MNnModel.mn_density)

    def evaluate_density_axis(self, r, axis):
        if axis == 'x':
            return self._evaluate_quantity(r, 0, 0, MNnModel.mn_density)
        if axis == 'y':
            return self._evaluate_quantity(0, r, 0, MNnModel.mn_density)
        else:
            return self._evaluate_quantity(0, 0, r, MNnModel.mn_density)

    def evaluate_forceR(self, x, y, z):
        """
        Returns the summed force of all the disks at a specific point. xyz can be scalars or a vector
        """
        return self._evaluate_quantity(x, y, z, MNnModel.mn_forceR)

    def evaluate_forceR_vec(self, x):
        """
        Returns the summed force of all the disks at a specific point. x is a Nx3 array
        """
        return self._evaluate_quantity(x[:,0], x[:,1], x[:,2], MNnModel.mn_forceR)

    def evaluate_forceV(self, x, y, z):
        """
        Returns the summed force of all the disks at a specific point. xyz can be scalars or a vector
        """
        return self._evaluate_quantity(x, y, z, MNnModel.mn_forceV)

    def evaluate_forceV_vec(self, x):
        """
        Returns the summed force of all the disks at a specific point. xyz can be scalars or a vector
        """
        return self._evaluate_quantity(x[0], x[1], x[2], MNnModel.mn_forceV)

    def evaluate_circular_velocity(self, x, y, z):
        
        return self._evaluate_quantity(x, y, z, MNnModel.mn_circular_velocity)        

    def is_positive_definite(self):
        """
        This function returns true if the models are positive definite.
        """
        mods = self.get_models()
        
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

            xopt, fval, ierr, nf = op.fminbound(self.evaluate_density_axis, 0.0, max_range, args = [axis], disp=0, full_output=True)
            if fval < 0.0:
                #print('Warning : This model has a root along the {0} axis (r={1}) : density can go below zero'.format(axis, x0))
                return False

        return True

    def generate_dataset_meshgrid(self, xmin, xmax, dx, quantity='density'):
        """
        Generates a meshgrid of data.
        :param xmin : a 3-tuple of floats indicating the low bound of the box
        :param xmax : a 3-tuple of floats indicating the high bound of the box
        :param dx : a 3-tuple of floats indicating the mesh spacing in every direction
        :param quantity : the type of quantity to fill the box with
        """

        quantity_vec = ('density', 'forceR', 'forceV', 'potential')
        if quantity not in quantity_vec:
            print('Error : Unknown quantity type {0}, possible values are {1}'.format(quantity, quantity_vec))
            return

        if len(xmin) != 3 or len(xmax) != 3 or len(dx) != 3:
            print('Error : You must provide xmin, xmax and dx as triplets of floats')
            return

        Xsp = []
        for i in range(3):
            Xsp.append(np.linspace(xmin[i], xmax[i], (xmax[i] - xmin[i]) / dx[i] + 1))

        try:
            gx, gy, gz = np.meshgrid(Xsp[0], Xsp[1], Xsp[2])
        except MemoryError:
            print('Error : The array you want to create is too big !')
            return

        if quantity == 'density':
            res = self.evaluate_density(gx, gy, gz)
        elif quantity == 'forceR':
            res = self.evaluate_forceR(gx, gy, gz)
        elif quantity == 'forceV':
            res = self.evaluate_forceV(gx, gy, gz)
        else:
            res = self.evaluate_potential(gx, gy, gz)
            
        return gx, gy, gz, res
        
        

    

    

    
