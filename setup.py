from distutils.core import setup

setup(
    # Application name:
    name="Multi Miyamoto Nagai",

    # Version number (initial):
    version="0.1.0",

    # Application author details:
    author="Armando Rojas",
    author_email="ozomatli@telmexmail.com",

    # Packages
    packages=['MultiMiyamotoNagai', 'MultiMiyamotoNagai.examples'],
    package_dir={'MultiMiyamotoNagai': 'MultiMiyamotoNagai', 'MultiMiyamotoNagai.examples': 'MultiMiyamotoNagai/examples'},
    package_data={'MultiMiyamotoNagai.examples': ['density.dat']},

    # Details
    url="http://www.github.com",

    # Requirements
    install_requires=['numpy', 'emcee', 'corner'],

    # Misc
    license="BSD",
    description="An analytical Multi Miyamoto Nagai model for galactic potential, density and forces evaluation.",
    #long_description=open("README.txt").read(),
)
