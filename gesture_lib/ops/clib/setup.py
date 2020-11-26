import numpy
from distutils.core import setup
from Cython.Build import cythonize
setup(
    ext_modules=cythonize("optimize.pyx"),
    include_dirs=[numpy.get_include()]
)
setup(
    ext_modules=cythonize("math.pyx")
)
