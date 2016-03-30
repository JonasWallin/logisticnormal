'''
Created on Aug 8, 2015
@author: Jonas Wallin
'''
from Cython.Build import cythonize
from numpy import get_include

try:
	from setuptools import setup, Extension
except ImportError:
	try:
		from setuptools.core import setup, Extension
	except ImportError:
		from distutils.core import setup, Extension
		
metadata = dict(
	  name='logisticnormal',
      version='0.1',
      author='Jonas Wallin',
      url='https://github.com/JonasWallin/logisticnormal',
      author_email='jonas.wallin81@gmail.com',
      requires=['numpy (>=1.3.0)',
                'cython (>=0.17)',
                'scipy'],
      #cmdclass={'build_ext': build_ext},
      packages=['logisticnormal', 'logisticnormal.PurePython',
                'logisticnormal.utils'],
      package_dir={'logisticnormal': 'logisticnormal/'},
      ext_modules=cythonize(
          [Extension("logisticnormal.distribution_cython",
                     sources=["logisticnormal/distribution_cython.pyx",
                              "logisticnormal/c/distribution_c.c"],
                     include_dirs=['.', get_include(), '/usr/include',
                                   '/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/'],
                     libraries=['gfortran', 'm', 'cblas', 'clapack'],
                     language='c'),
           Extension("logisticnormal.logisticNormal",
                     sources=["logisticnormal/logisticNormal.pyx",
                              "logisticnormal/c/distribution_c.c"],
                     include_dirs=['.', get_include(), '/usr/include',
                                   '/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/'],
                     libraries=['gfortran', 'm', 'cblas', 'clapack'],
                     language='c')]),
              )
setup(**metadata)
