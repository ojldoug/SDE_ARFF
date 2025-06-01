from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="gillespie",
    ext_modules=cythonize(
        Extension(
            "gillespie_c",  # Name of the extension
            sources=["gillespie_c.c"],  # Only include this source file once
            include_dirs=[numpy.get_include()],  # Include NumPy headers
        )
    ),
)
