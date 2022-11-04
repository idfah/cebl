#!/usr/bin/env python

import numpy as np
import sys
import subprocess

import setuptools
from Cython.Build import cythonize

# c extension modules
c_modules = []

extra_compile_args = ['-Wall',] #'-march=native', '-O3']

# these devices are currently only supported on linux
if sys.platform.startswith('linux'):

    # Neuropulse Mindset-24R
    c_modules.append(
        setuptools.Extension('cebl.rt.sources.neuropulse.libmindset24r',
            sources=['cebl/rt/sources/neuropulse/libmindset24r.c'],
            extra_compile_args=extra_compile_args))

    # BioSemi ActiveTwo
    c_modules.append(
        setuptools.Extension('cebl.rt.sources.biosemi.libactivetwo',
            sources=['cebl/rt/sources/biosemi/activetwo.c'],
            libraries=['bsif', 'usb-1.0'],
            #libraries=['usb'],
            library_dirs=['cebl/rt/sources/biosemi/'],
            language='c++',
            extra_compile_args=extra_compile_args))

    # fast tanh in c
    ## c_modules.append(
    ##     setuptools.Extension('cebl.util.fasttanh',
    ##         sources=['cebl/util/fasttanh.c'],
    ##         libraries=['pthread', 'gomp'],
    ##         include_dirs=[np.get_include()],
    ##         extra_compile_args=extra_compile_args + ['-fopenmp',]))

# cython extension modules
cython_modules = []

# source extension
cython_modules.append(
    setuptools.Extension('cebl.rt.sources.source.source',
        sources=['cebl/rt/sources/source/source.pyx'],
        extra_compile_args=extra_compile_args))

# cythonized wx.lib.plot
cython_modules.append(
    setuptools.Extension('cebl.rt.widgets.wxlibplot.plotcanvas',
        sources=['cebl/rt/widgets/wxlibplot/plotcanvas.pyx'],
        extra_compile_args=extra_compile_args))

cython_modules.append(
    setuptools.Extension('cebl.rt.widgets.wxlibplot.polyobjects',
        sources=['cebl/rt/widgets/wxlibplot/polyobjects.pyx'],
        extra_compile_args=extra_compile_args))

## # g.tec g.MOBILab+
## cython_modules.append(
##     setuptools.Extension('cebl.rt.sources.gtec.gmobilab.gmobilab',
##     sources=['cebl/rt/sources/gtec/gmobilab/gmobilab.pyx'],
##     extra_compile_args=extra_compile_args))
##
## # g.tec g.Nautilus
## cython_modules.append(
##     setuptools.Extension('cebl.rt.sources.gtec.gnautilus.gnautilus',
##     sources=['cebl/rt/sources/gtec/gnautilus/gnautilus.pyx'],
##     extra_compile_args=extra_compile_args))

# all extension modules
ext_modules=c_modules + cythonize(cython_modules, language_level='3')

# extract version from startup script
# this is all hacky - XXX idfah
version=subprocess.check_output(['scripts/cebl', '--version']).decode()
version='.'.join(version.split('.')[:3])

setuptools.setup(
    name='CEBL',
    version=version,
    author='Elliott Forney and Charles Anderson',
    author_email='eeg@cs.colostate.edu',
    url='http://www.cs.colostate.edu/eeg',
    packages=setuptools.find_packages(),
    ext_modules=ext_modules,
    scripts=['scripts/cebl'],
    license='GPL3, Copyright (2017) Elliott Forney, Charles Anderson, Colorado State University',
    description='Colorado Electroencephalography and Brain-Computer Interfaces Laboratory (CEBL)',
    include_package_data=True,
    install_requires=['matplotlib', 'numpy', 'scipy', 'wxPython', 'pylibftdi', 'serial']
)
