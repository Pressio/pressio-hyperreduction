#!/usr/bin/env python
# Authors:
# Francesco Rizzi (fnrizzi@sandia.gov, francesco.rizzi@ng-analytics.com)
# Patrick Blonigan (pblonig@sandia.gov)
# Eric Parish (ejparis@sandia.gov)
# John Tencer (jtencer@sandia.gov)

import os
import sys
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

topdir = os.path.abspath(os.path.dirname(__file__))

def description():
    with open(os.path.join(topdir, 'DESCRIPTION.rst')) as f:
        return f.read()

# A CMakeExtension needs a sourcedir instead of a file list.
class CMakeExtension(Extension):
  def __init__(self, name, sourcedir=""):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
  def build_extension(self, ext):
    extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

    if not extdir.endswith(os.path.sep): extdir += os.path.sep

    if not os.path.exists(self.build_temp): os.makedirs(self.build_temp)
    print("self.build_temp ", self.build_temp)

    # CMake lets you override the generator - we need to check this.
    # Can be set with Conda-Build, for example.
    cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

    #------------------------
    cfg = "Debug" if self.debug else "Release"

    if "CXX" not in os.environ:
      msg = "CXX env var missing, needs to point to your mpic++ compiler"
      raise RuntimeError(msg)

    if "TRILINOS_ROOT" not in os.environ:
      msg = "TRILINOS_ROOT env var missing, needs to point to your Trilinos install"
      raise RuntimeError(msg)

    # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
    cmake_args = [
      "-DCMAKE_VERBOSE_MAKEFILE={}".format("ON"),
      "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
      "-DPYTHON_EXECUTABLE={}".format(sys.executable),
      "-DCMAKE_BUILD_TYPE={}".format(cfg),
      "-DTRILINOS_ROOT={}".format(os.environ.get("TRILINOS_ROOT")),
      "-DVERSION_INFO={}".format(self.distribution.get_version()),
    ]
    build_args = []

    # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
    # across all generators.
    if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
      # for now, set parallel level to 4
      build_args += ["-j4"]
      # # self.parallel is a Python 3 only way to set parallel jobs by hand
      # # using -j in the build_ext call, not supported by pip or PyPA-build.
      # if hasattr(self, "parallel") and self.parallel:
      #   # CMake 3.12+ only.
      #   build_args += ["-j{}".format(self.parallel)]

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    subprocess.check_call(
      ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
    )
    subprocess.check_call(
      ["cmake", "--build", "."] + build_args, cwd=self.build_temp
    )

setup(
  name="pressio-tools",
  version="0.6.1rc1",
  author="TBD",
  author_email="TBD",
  description="pressio-tools",
  #
  project_urls={
    'Source': 'https://github.com/Pressio/pressio-tools'
  },
  #
  long_description=description(),
  ext_modules=[CMakeExtension("pressio-tools")],
  cmdclass={"build_ext": CMakeBuild},
  install_requires=["numpy", "scipy", "matplotlib", "mpi4py"],
  zip_safe=False,
  #
  python_requires='>=3',
  classifiers=[
    "License :: OSI Approved :: BSD License",
    "Operating System :: Unix",
    "Environment :: MacOS X",
    "Programming Language :: C++",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Software Development :: Libraries",
    "Development Status :: 4 - Beta"
  ],
  keywords=["model reduction", "scientific computing", "dense linear algebra", "HPC"],
)
