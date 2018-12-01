# from Cython.Build import cythonize
# import numpy as np
# from distutils.core import setup
#
# try:
#     numpy_include = np.get_include()
# except AttributeError:
#     numpy_include = np.get_numpy_include()
#
# setup(
#     ext_modules=cythonize(["bbox.pyx", "cython_nms.pyx"], include_dirs=[numpy_include])
# )


# 先执行

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("bbox", ["bbox.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)

# setup(
#     ext_modules=[
#         Extension("cython_nms", ["cython_nms.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )

# 执行 python setup_other.py build