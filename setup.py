try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


packages = [
    'priorCVAE',
    'priorCVAE.datasets',
    'priorCVAE.losses',
    'priorCVAE.models',
    'priorCVAE.priors'
 ]

setup(name='priorCVAE',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      version='1.0',
      description='Prior CVAE',
      author='MLGH',
      packages=packages
      )
