
import argparse, sys, importlib
import importlib
from pressiotools.io.yaml_parser import yaml_read
from pressiotools.samplemesh.galerkinProjector import computeGalerkinProjector

#--------------------------
if __name__ == '__main__':
#--------------------------
  # check if mpi4py module is present, if so import it
  mpi4pyspec = importlib.util.find_spec("mpi4py")
  mpi4pyfound = mpi4pyspec is not None
  if mpi4pyfound:
    # mpi4py initializes mpi when the module is imported so
    # it makes sense to put the import here
    from mpi4py import MPI

  descString = "Create Galerkin projector matrix needed by \
hyper-reduced Galerkin ROMs in pressio."

  #------ cmd line args -------#
  parser = argparse.ArgumentParser(description=descString)
  parser.add_argument(
    '--input', '-input', '-i', '--i',
    help="full path to yaml input file.",
    required=True)

  parser.add_argument(
    '--data-dir', '-data-dir', '-d', '--d',
    dest="dataDir",
    help="full path to where data lives. Data includes state basis, residual basis, \
         sample mesh indices lives.",
    required=True)

  parser.add_argument(
    '--out-dir', '-out-dir', '-o', '--o',
    dest="outDir", default="",
    help="full path to directory to store results, i.e. file with the projector matrix\
If nothing is passed, we use as output the directory where data is loaded from.",
    required=False)
  args = parser.parse_args()

  #------ process args -------#
  # read yaml file
  yaml_in = yaml_read(args.input)

  # insert in dic where data lives
  yaml_in["dataDir"] = args.dataDir

  # set the output directory
  if args.outDir == "":
    yaml_in["ProjectorMatrix"]["outDir"] = args.dataDir
  else:
    yaml_in["ProjectorMatrix"]["outDir"] = args.outDir

  # if mpi4py is found, set commworld otherwise nullify comm
  # this is needed in the implementation to handle the null comm
  comm = MPI.COMM_WORLD if mpi4pyfound else None

  # compute projector and write to file
  computeGalerkinProjector(communicator=comm, yamldic=yaml_in, fromDriver=True)
