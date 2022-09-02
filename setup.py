from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

"""
shell command: 
python setup.py install

If access is denied:
python setup.py install --user 
"""

nom="pyHSSMsol" #name of the pyx file to compile and also the name of the package

extensions=[
	Extension(
		nom,
		[nom+".pyx"],
		include_dirs = [numpy.get_include()],
	)
]

setup(
    name=nom,
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}),
)

print("\n")
print("###############################")
print("# Model successfully compiled #")
print("###############################")
