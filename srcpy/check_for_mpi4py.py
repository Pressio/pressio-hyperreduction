
import importlib

def importMpiIfNeeded():
  mpi4pyspec = importlib.util.find_spec("mpi4py")
  mpi4pyfound = mpi4pyspec is not None
  if mpi4pyfound:
    # mpi4py initializes mpi when the module is imported so
    # it makes sense to put the import here
    from mpi4py import MPI
