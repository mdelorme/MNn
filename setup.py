from distutils.core import setup

setup(
    # Application name:
    name="MNn",

    # Version number (initial):
    version="0.1.0",

    # Application author details:
    author="Armando Rojas",
    author_email="ozomatli@telmexmail.com",

    # Packages
    packages=['mnn', 'mnn.examples'],
    package_dir={'mnn': 'mnn', 'mnn.examples': 'mnn/examples'},
    package_data={'mnn.examples': ['density.dat']},

    # Details
    url="http://www.github.com/mdelorme/MNn",

    # Requirements
    install_requires=['numpy', 'emcee', 'corner'],

    # Misc
    license="BSD",
    description="An analytical sum of Miyamoto-Nagai-negative model for galactic potential, density and forces evaluation.",
    #long_description=open("README.txt").read(),
)
