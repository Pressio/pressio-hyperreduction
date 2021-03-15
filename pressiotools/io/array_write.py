
import numpy as np
import math

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
  if isinstance(arr, np.ndarray):
    write_array(arr, myFileName, isBinary)
  else:
    myArr = arr.data()
    write_array(myArr, myFileName, isBinary)
