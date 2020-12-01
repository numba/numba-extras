# import os
# import platform
# import sys
# from distutils import sysconfig
# from distutils.command import build
# from distutils.command.build_ext import build_ext
# from distutils.spawn import spawn
from setuptools import find_packages, setup
# import versioneer
# versioneer.VCS = 'git'
# versioneer.versionfile_source = 'numba/_version.py'
# versioneer.versionfile_build = 'numba/_version.py'
# versioneer.tag_prefix = ''
# versioneer.parentdir_prefix = 'numba-'

# cmdclass = versioneer.get_cmdclass()
# cmdclass['build_doc'] = build_doc


packages = find_packages(include=["numba_extras", "numba_extras.*"])

# build_requires = ['numpy >={}'.format(min_numpy_build_version)]
# install_requires = [
#     'llvmlite >={},<{}'.format(min_llvmlite_version, max_llvmlite_version),
#     'numpy >={}'.format(min_numpy_run_version),
#     'setuptools',
# ]

metadata = dict(
    name='numba',
    description="compiling Python code using LLVM",
    # version=versioneer.get_version(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Compilers",
    ],
    # package_data={
    #     # HTML templates for type annotations
    #     "numba.core.annotations": ["*.html"],
    #     # Various test data
    #     "numba.cuda.tests.cudadrv.data": ["*.ptx"],
    #     "numba.tests": ["pycc_distutils_usecase/*.py"],
    #     # Some C files are needed by pycc
    #     "numba": ["*.c", "*.h"],
    #     "numba.pycc": ["*.c", "*.h"],
    #     "numba.core.runtime": ["*.c", "*.h"],
    #     "numba.cext": ["*.c", "*.h"],
    #     # numba gdb hook init command language file
    #     "numba.misc": ["cmdlang.gdb"],
    # },
    # scripts=["numba/pycc/pycc", "bin/numba"],
    author="Anaconda, Inc.",
    # author_email="numba-users@continuum.io",
    url="https://numba.github.com",
    packages=packages,
    # setup_requires=build_requires,
    # install_requires=install_requires,
    # python_requires=">={},<{}".format(min_python_version, max_python_version),
    license="BSD",
    # cmdclass=cmdclass,
)

# with open('README.rst') as f:
#     metadata['long_description'] = f.read()

setup(**metadata)
