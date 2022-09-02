from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


"""
compilateur: Microsoft Visual C++ 2019 (MSVC) installé avec la version community de visual studio 2019
version de python: Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 22:22:05) [MSC v.1916 64 bit (AMD64)] on win32

Installation sur le pc d'un module cython

ouvrez un shell widows (C:\Windows\System32\cmd.exe)
Allez jusqu'au répertoire du ficier pyx
tapez dans le shell: python setup.py install
Si l'accès est refusé tapez:
python setup.py install --user 
"""

nom="pyHSSMsol" #nom du fichier pyx à compiler et également le nom du module une fois complilé

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
