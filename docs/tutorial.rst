Tutorial
========

This tutorial will help you learn how to manipulate and use ``MNn`` to build models, fit them to data, and use them to retrieve potentials and densities.
The contents of this section are based on the examples you can find in the ``example`` folder of the code, or on the `repository <https://github.com/mdelorme/MNn/tree/master/mnn/examples>`_.

Simple model
------------

First and foremost, let's create a simple model with only one Miyamoto Nagai disc.
The disc will be aligned with the $xy$ plane (thus the axis will be ``z``), and will have the following parameters : ``a=1.0``, ``b=10.0``, ``M=100.0``.

The first step is to import and instantiate the :class:`~mnn.model.MNnModel` class that represents our model :

>>> from mnn.model import MNnModel
>>> model = MNnModel()

Now we can add a Miyamoto-Nagai disc to the model :

>>> model.add_disc('z', 1.0, 10.0, 100.0)

The model is now ready to use. We can, for instance, retrieve the density and the potential of the model at cartesian coordinates ``(1.0, 2.0, -0.5)`` :
    
>>> model.evaluate_potential(1.0, 2.0, -0.5)
-0.038273018555213874

>>> model.evaluate_density(1.0, 2.0, -0.5)
0.016676480491325717

These methods can also be used with vectors. For instance, to evaluate the density along the x-axis, we can do :

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> x = np.linspace(0.0, 30.0, 100.0)
>>> density = model.evaluate_density(x, 0.0, 0.0)
>>> plt.plot(x, density)
>>> plt.show()

You should obtain the following result :

.. image:: images/tut_density1.png
  
We can also use the :func:`~mnn.model.MNnModel.evaluate_density_vec` method of :class:`~mnn.model.MNnModel` to group the ``x``, ``y``, and ``z`` values in a single numpy array. For instance to evaluate potential over two points ``(1.0, 2.0, -0.5)`` and ``(-5.0, 0.0, 0.0)`` we can use :

>>> p = np.array(((2.0, 1.0, -0.5), (-5.0, 0.0, 0.0)))
>>> model.evaluate_potential_vec(p)
array([-0.03827302, -0.03559385])

Once the model is completed, you can generate a meshgrid of data. For instance, let's generate and plot a slice of the xz plane at ``y=0``, and for x in $[0, 30]$ and z in $[-10, 10]$. To do this, we need to define a mesh size. We will make cells $0.1$ units wide.

>>> x, y, z, v = model.generate_dataset_meshgrid((0.0, 0.0, -10.0), (30.0, 0.0, 10.0), (0.1, 0.1, 0.1))
>>> plt.imshow(v[0].T)
>>> plt.show()

Should give you :

.. image:: images/tut_density_mesh1.png

.. note:: Please note that the scales on the image are wrong here because the values have been plotted without mention to the ``x`` and ``z`` tables.
	  This is mainly due for the tutorial briefness.

.. note:: By default, the :func:`~mnn.model.MNnModel.generate_dataset_meshgrid` method generates the density.
	  But you can ask it to generate the potential by using the keyword ``quantity='potential'``
		    
	  
Finally, we can plot contour lines at the cost of a little more effort :

>>> x = np.arange(0.0, 30.1, 0.1)
>>> z = np.arange(-10.0, 10.1, 0.1)
>>> plt.contour(x, z, v[0].T)
>>> plt.show()

These commands should give the following contour plot :

.. image:: images/tut_density_contour1.png


Multiple discs and negative scales
----------------------------------

The strength of ``MNn`` is to provide a model that sums multiple Miyamoto-Nagai discs, and that some of these models can have a negative disc scale (``a``). Let's create such a model with three discs. for this we can use the previous method :func:`~mnn.model.MNnModel.add_disc` or use a the wrapper :func:`~mnn.model.MNnModel.add_discs`. This wrapper takes a list of discs as we would create them with :func:`~mnn.model.MNnModel.add_disc`.

>>> model = MNnModel()
>>> discs = (('z', 10.0, 10.0, 100.0), ('y', -7.0, 20.0, 10.0))
>>> model.add_discs(discs)

.. note:: The discs can have ``a<0`` as long as ``a+b>=0``. The other constraints on the model are : ``b>=0`` and ``M>=0``.

This new model can be used as previously, for instance plotting the density on the ``x=0`` plane :

>>> x, y, z, v = model.generate_dataset_meshgrid((0.0, -20.0, -20.0), (0.0, 20.0, 20.0), (0.1, 0.1, 0.1))
>>> plt.imshow(v[:, 0].T)
>>> plt.show()

Will give you :

.. image:: images/tut_density_mesh2.png

And :

>>> y = np.arange(-30.0, 30.1, 0.1)
>>> z = np.arange(-30.0, 30.1, 0.1)
>>> plt.contour(y, z, v[:, 0].T)
>>> plt.show()

Will yield :

.. image:: images/tut_density_contour2.png
