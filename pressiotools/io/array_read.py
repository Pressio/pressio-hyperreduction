
import numpy as np
import math
from pressiotools import linalg as la

def read_binary_array(fileName, nCols):
  # read a numpy array from a binary file "fileName"
  if nCols==1:
    return np.fromfile(fileName)
  else:
    array = np.fromfile(fileName)
    nRows = int(len(array) / float(nCols))
    return array.reshape((nCols,nRows)).T

def read_ascii_array(fileName, nCols):
  # read a numpy array from an ascii file "fileName"
  return np.asfortranarray(np.loadtxt(fileName))

def read_array(fileName, nCols, isBinary=True):
  if isBinary:
    return read_binary_array(fileName,nCols)
  else:
    return read_ascii_array(fileName,nCols)

def read_array_distributed(comm, rootFileName, nCols, isBinary=True):
  # Read an array from binary or ascii files with the name specified
  # by the string rootFileName
  # Each local array segment will be read from a file rootFileName.XX.YY,
  # where XX is the number of ranks and YY is the local rank
  rank = comm.Get_rank()
  size = comm.Get_size()

  nDigit = int(math.log10(size)) + 1
  myFileName = "{}.{}.{:0{width}d}".format(rootFileName,size,rank,width=nDigit)
  myArr = read_array(myFileName,nCols,isBinary)

  if nCols==1:
    return la.Vector(myArr)
  else:
    return la.MultiVector(myArr)
