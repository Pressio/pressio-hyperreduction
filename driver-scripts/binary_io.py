import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../..")

import numpy as np
from mpi4py import MPI
import pressiotools as pt
import math

def read_binary_array(fileName,nCols):
  # read a numpy array from a binary file "fileName".bin
  if nCols==1:
    return np.fromfile(fileName)
  else:
    array = np.fromfile(fileName)
    nRows = int(len(array) / float(nCols))
    return array.reshape((nCols,nRows)).T

def read_binary_array_distributed(fileName,nCols):
  # Read an array from binary files with the name specified by the string fileName
  # Each local array segment will be written to a file fileName.XX.YY, 
  # where XX is the number of ranks and YY is the local rank
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  nDigit = int(math.log10(size)) + 1

  # write BaseVector portion on each processor
  myFileName = "{}.{}.{:0{width}d}".format(fileName,size,rank,width=nDigit)
  myArr = read_binary_array(myFileName,nCols)
  
  if nCols==1:
    return pt.Vector(myArr)    
  else:
    return pt.MultiVector(myArr)

def write_binary_array(arr,fileName):
  # write numpy array arr to a binary file "fileName"
  f = open(fileName,'w')
  arr.T.tofile(f)
  f.close()

def write_binary_array_distributed(arr,fileName):
  # Write array arr to binary files with the name specified by the string fileName
  # Each local array segment will be written to a file fileName.XX.YY, 
  # where XX is the number of ranks and YY is the local rank
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  nDigit = int(math.log10(size)) + 1

  # write BaseVector portion on each processor
  f = open("{}.{}.{:0{width}d}".format(fileName,size,rank,width=nDigit),'w')
  myArr = arr.data()
  myArr.T.tofile(f)
  f.close()

