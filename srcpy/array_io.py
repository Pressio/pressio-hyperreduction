
import numpy as np
import math
import pressiotools as pt

# --------------------
# reading functions
# --------------------
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
    return pt.Vector(myArr)
  else:
    return pt.MultiVector(myArr)

# --------------------
# writing functions
# --------------------
def write_binary_array(arr, fileName):
  # write numpy array arr to a binary file "fileName"
  f = open(fileName,'w')
  arr.T.tofile(f)
  f.close()

def write_ascii_array(arr, fileName):
  # write numpy array arr to an ascii file "fileName"
  np.savetxt(fileName,arr)

def write_array(arr, fileName, isBinary=True):
  if isBinary:
    return write_binary_array(arr,fileName)
  else:
    return write_ascii_array(arr,fileName)

def write_array_distributed(comm, arr, rootFileName, isBinary=True):
  # Write array arr to binary or ascii files with the name specified by the string rootFileName
  # Each local array segment will be written to a file rootFileName.XX.YY,
  # where XX is the number of ranks and YY is the local rank
  rank = comm.Get_rank()
  size = comm.Get_size()

  nDigit = int(math.log10(size)) + 1
  # write BaseVector portion on each processor
  myFileName = "{}.{}.{:0{width}d}".format(rootFileName,size,rank,width=nDigit)
  myArr = arr.data()
  write_array(myArr,myFileName,isBinary)
