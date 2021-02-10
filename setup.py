#!/usr/bin/env python
# Authors:
# Francesco Rizzi (francesco.rizzi@ng-analytics.com)
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

    cfg = "Debug" if self.debug else "Release"

    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    #-------------------------------------------------------------------
    # check that MPI_BASE_DIR is in the environment
    if "MPI_BASE_DIR" not in os.environ:
      msg = "\n **ERROR**: \n MPI_BASE_DIR env var is missing, needs to point to your MPI installation directory "
      raise RuntimeError(msg)

    #-------------------------------------------------------------------
    # TRILINOS
    #-------------------------------------------------------------------
    # check if TRILINOS_ROOT is present, if not attemp build
    trilroot = os.environ.get("TRILINOS_ROOT")
    print(trilroot)
    if "TRILINOS_ROOT" not in os.environ or trilroot == "":
      msg = "TRILINOS_ROOT not found, attempting to build"

      trilTarName = "trilinos-release-13-0-1.tar.gz"
      trilUnpackedName = "Trilinos-trilinos-release-13-0-1"
      trilUrl = "https://github.com/trilinos/Trilinos/archive/"+trilTarName

      cwd = os.getcwd()

      # create subdirs for trilinos
      trilinosSubDir = cwd + "/"+self.build_temp+"/../trilinos"
      if not os.path.exists(trilinosSubDir): os.makedirs(trilinosSubDir)

      trilTarPath = trilinosSubDir+"/"+trilTarName
      print("trilTarPath ", trilTarPath)
      if not os.path.exists(trilTarPath):
        subprocess.check_call(
          ["wget", "--no-check-certificate", trilUrl], cwd=trilinosSubDir
        )

      trilSrcDir = trilinosSubDir+"/"+trilUnpackedName
      print("trilSrcPath ", trilSrcDir)
      if not os.path.exists(trilSrcDir):
        subprocess.check_call(
          ["tar", "zxf", trilTarName], cwd=trilinosSubDir
        )

      trilBuildDir = trilinosSubDir+"/build"
      print("trilBuildDir = ", trilBuildDir)
      trilInstallDir = trilinosSubDir+"/install"
      print("trilInstall = ", trilInstallDir)

      cmake_args = [
        "-DCMAKE_BUILD_TYPE={}".format("Release"),
        "-DBUILD_SHARED_LIBS={}".format("ON"),
        "-DCMAKE_VERBOSE_MAKEFILE={}".format("ON"),
        "-DTPL_ENABLE_MPI={}".format("ON"),
        "-DMPI_BASE_DIR={}".format(os.environ.get("MPI_BASE_DIR")),
        "-DTrilinos_ENABLE_Tpetra={}".format("ON"),
        "-DTrilinos_ENABLE_TpetraTSQR={}".format("ON"),
        "-DTrilinos_ENABLE_Epetra={}".format("ON"),
        "-DTrilinos_ENABLE_Ifpack={}".format("ON"),
        "-DTrilinos_ENABLE_Ifpack2={}".format("ON"),
        "-DTrilinos_ENABLE_Triutils={}".format("ON"),
        "-DCMAKE_INSTALL_PREFIX={}".format(trilInstallDir),
      ]

      if not os.path.exists(trilBuildDir):
        os.makedirs(trilBuildDir)

        subprocess.check_call(
          ["cmake", trilSrcDir] + cmake_args, cwd=trilBuildDir
        )
        subprocess.check_call(
          ["cmake", "--build", ".", "-j4"], cwd=trilBuildDir
        )
        subprocess.check_call(
          ["cmake", "--install", "."], cwd=trilBuildDir
        )

      # set env var
      os.environ["TRILINOS_ROOT"] = trilInstallDir

    else:
      msg = "Found env var TRILINOS_ROOT={}".format(os.environ.get("TRILINOS_ROOT"))

    #-------------------------------------------------------------------
    # build/install pressio-tools
    #-------------------------------------------------------------------
    cc = os.environ.get("MPI_BASE_DIR")+"/bin/mpicc"
    cxx = os.environ.get("MPI_BASE_DIR")+"/bin/mpicxx"
    cmake_args = [
      "-DCMAKE_C_COMPILER={}".format(cc),
      "-DCMAKE_CXX_COMPILER={}".format(cxx),
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

    subprocess.check_call(
      ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
    )
    subprocess.check_call(
      ["cmake", "--build", "."] + build_args, cwd=self.build_temp
    )

setup(
  name="pressio-tools",
  version="0.6.1rc1",
  author="F.Rizzi,P.Blonigan,E.Parish",
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
  install_requires=["numpy", "scipy", "mpi4py", "pyyaml"],
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
