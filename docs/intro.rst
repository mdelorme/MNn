Introduction
============

Pre-requisites
--------------

``MNn`` relies on multiple python packages. These are hard dependencies that need to be installed :

* `numpy  <http://www.numpy.org/>`_ and `scipy <http://www.scipy.org>`_
* `matplotlib <http://matplotlib.org/>`_
* `emcee <http://dan.iel.fm/emcee/current/>`_
* `corner <https://github.com/dfm/corner.py>`_

  The package has been tested for Python 2.7.8+. It should work regardless of the version of Python you are using, as long as you can find a compatible
  version of every library.
  
Installation
------------

The only way to install ``MNn`` is from the source. The package is hosted on `github <https://github.com/mdelorme/MNn>`_.
To download the code to a folder use the following command ::

  git clone https://github.com/mdelorme/MNn

Then install it to your system ::

  cd MNn
  python setup.py install


Testing the installation
------------------------

In the source folder ::

  cd mnn/examples
  python simple_model.py

You should obtain the following result ::
    
0.0166764804913
-0.0382730185552

As well as three figures that are explained in the :doc:`tutorial` section.
