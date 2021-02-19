
import sys
import numpy as np
from pressiotools.io.array_read import *
from pressiotools.io.array_write import *
from pressiotools import linalg as ptla

#-----------------------------------------------------------------
def _processYamlDictionary(yamlDic):
  dataDir      = yamlDic["dataDir"]
  outDir       = yamlDic["outDir"]
  fileRootName = yamlDic["SVD"]["file-root-name"]
  fileFmt      = yamlDic["SVD"]["format"]
  nCols        = yamlDic["SVD"]["num-columns"]

  if fileFmt=="binary":
    isBinary=True
  elif fileFmt=="ascii":
    isBinary=False
  else:
    sys.exit("Unsupported file format: matrix for svd must be 'binary' or 'ascii'.")

  return {"dataDir"        : dataDir,
          "outDir"         : outDir,
          "fileRootName"   : fileRootName,
          "isBinary"       : isBinary,
          "nCols"          : nCols,
          'leftSingularVectors' : yamlDic['SVD']['leftSingularVectors']}

#-----------------------------------------------------------------
def _readData(dic, comm=None):
  dataDir      = dic["dataDir"]
  rootFileName = dic["fileRootName"]
  nCols        = dic["nCols"]
  isBinary     = dic["isBinary"]
  if comm is not None and comm.Get_size() > 1:
    fileName = dataDir + "/" + rootFileName
    psi = read_array_distributed(comm, fileName, nCols, isBinary)
    return psi
  else:
    fileName = "{}.{}.{:0{width}d}".format(rootFileName,1,0,width=1)
    psi0 = read_array(dataDir + "/" + fileName, nCols, isBinary)
    psi  = ptla.MultiVector(psi0)
    return psi

#-----------------------------------------------------------------
def _computeSvdYamlInput(comm, yaml_in):
  nullComm = True if comm is None else False
  rank     = 0 if nullComm else comm.Get_rank()
  nRanks   = 1 if nullComm else comm.Get_size()
  if rank == 0: print("Processing yaml file and computing SVD")

  # process user inputs
  dic = _processYamlDictionary(yaml_in)

  svecNode = dic['leftSingularVectors']
  svecWriteFileName = svecNode['file-root-name']
  svecWriteFormat   = svecNode['write-format']
  svecNumToKeep     = svecNode['how-many']

  # # check that inputs are valid, exit with error otherwise
  # _checkInputsValidity(dic)

  # read data matrix
  A = _readData(dic, comm)

  svdObject = ptla.Svd()
  svdObject.computeThin(A)
  U  = svdObject.viewLeftSingVectorsLocal()
  S  = svdObject.viewSingValues()
  VT = svdObject.viewRightSingVectorsT()
  if rank == 0:
    print("Computation completed, outputting results")

  Ufile = dic['outDir']+"/"+svecWriteFileName+".txt"+str(rank)
  np.savetxt(Ufile, U[:, :svecNumToKeep], fmt="%.15f")
  Sfile = dic['outDir']+"/singularValues.txt"
  if rank == 0:
    np.savetxt(Sfile, S, fmt="%.15f")

  if rank == 0: print("All done!")

#-----------------------------------------------------------------
# this is the entry function visibile outside, all the other functions
# above are implementation details (note the _ prepending all names)
# and should not be exposed outside
#-----------------------------------------------------------------
def computeSvd(**args):
  if len(args) == 2 \
     and 'communicator' in args.keys() \
     and 'yamldic'      in args.keys():
    _computeSvdYamlInput(args['communicator'], args['yamldic'])

  else:
    sys.exit("svd currently only supported via yaml file")
