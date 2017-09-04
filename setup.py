#!/usr/bin/env python

#import distutils.core as dc
import setuptools
from Cython.Build import cythonize
import numpy as np


cebl_rt_sources_source_ext = setuptools.Extension('cebl.rt.sources.source.source',
    sources = ['cebl/rt/sources/source/source.pyx'],
    extra_compile_args = ['-march=core2', '-O3'])

cebl_rt_sources_neuropulse_m24rlib_ext = setuptools.Extension('cebl.rt.sources.neuropulse.libmindset24r',
    sources = ['cebl/rt/sources/neuropulse/libmindset24r.c'],
    extra_compile_args = ['-Wall'])

cebl_rt_sources_biosemi_activetwolib_ext = setuptools.Extension('cebl.rt.sources.biosemi.libactivetwo',
    sources = ['cebl/rt/sources/biosemi/activetwo.c'],
    libraries = ['bsif', 'usb'],
    library_dirs = ['cebl/rt/sources/biosemi/'],
    extra_compile_args = ['-Wall', '-march=core2', '-O3'],
    language='c++')

cebl_rt_widgets_wxlibplot_ext = setuptools.Extension('cebl.rt.widgets.wxlibplot',
    sources = ['cebl/rt/widgets/wxlibplot.pyx'],
    extra_compile_args = ['-march=core2', '-O3'])

cebl_sig_cwt_ext = setuptools.Extension('cebl.sig.cwt',
    sources = ['cebl/sig/cwt.pyx'],
    libraries = ['pthread', 'gomp'],
    extra_compile_args = ['-march=core2', '-O3'])

cebl_util_fasttanh_ext = setuptools.Extension('cebl.util.fasttanh',
    sources = ['cebl/util/fasttanh.c'],
    libraries = ['pthread', 'gomp'],
    include_dirs = [np.get_include()],
    extra_compile_args = ['-march=core2', '-O3', '-fopenmp'])


ext_modules = [cebl_rt_sources_neuropulse_m24rlib_ext,
               cebl_rt_sources_biosemi_activetwolib_ext,
               cebl_util_fasttanh_ext] + \
               cythonize([cebl_rt_sources_source_ext,
                          cebl_rt_widgets_wxlibplot_ext,
                          cebl_sig_cwt_ext])


setuptools.setup(
    name='CEBL',
    version='3.0.0',
    author='Elliott Forney and Charles Anderson',
    author_email='eeg@cs.colostate.edu',
    url='http://www.cs.colostate.edu/eeg',
    packages=setuptools.find_packages(),
    ext_modules = ext_modules,
    scripts=['scripts/cebl'],
    license='GPL3, Copyright (2017) Elliott Forney, Charles Anderson, Colorado State University',
    description='Colorado Electroencephalography and Brain-Computer Interfaces Laboratory (CEBL)',
    include_package_data=True,
    #install_requires=['matplotlib', 'scipy', 'numpy', 'pylibftdi']
)
