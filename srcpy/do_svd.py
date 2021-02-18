
import numpy as np
import pathlib, sys
import pressiotools as pt
import array_io

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
    psi = array_io.read_array_distributed(comm, fileName, nCols, isBinary)
    return psi
  else:
    fileName = "{}.{}.{:0{width}d}".format(rootFileName,1,0,width=1)
    psi0 = array_io.read_array(dataDir + "/" + fileName, nCols, isBinary)
    psi  = pt.MultiVector(psi0)
    return psi

#-----------------------------------------------------------------
def _computeSvdReadYaml(comm, yaml_in):
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
  #print(A.data())

  # compute
  if nullComm or nRanks==1:
    import scipy.linalg.lapack as la
    U,S,V = np.linalg.svd(A.data(), full_matrices=False)
    #np.savetxt(outDir+"/"+svecWriteFileName".txt", mySampleMeshNodes, fmt="%d")
    print("Computation completed, outputting results")

    Ufile = dic['outDir']+"/"+svecWriteFileName+".txt"+str(0)
    np.savetxt(Ufile, U[:, :svecNumToKeep], fmt="%.15f")

    Sfile = dic['outDir']+"/singularValues.txt"
    if rank == 0: np.savetxt(Sfile, S, fmt="%.15f")

  else:
    svdObject = pt.svd()
    svdObject.computeThin(A)
    U  = svdObject.viewLeftSingVectorsLocal()
    S  = svdObject.viewSingValues()
    VT = svdObject.viewRightSingVectorsT()

    if rank == 0: print("Computation completed, outputting results")

    Ufile = dic['outDir']+"/"+svecWriteFileName+".txt"+str(rank)
    np.savetxt(Ufile, U[:, :svecNumToKeep], fmt="%.15f")

    Sfile = dic['outDir']+"/singularValues.txt"
    if rank == 0: np.savetxt(Sfile, S, fmt="%.15f")

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
    _computeSvdReadYaml(args['communicator'], args['yamldic'])

  else:
    sys.exit("svd currently only supported via yaml file")
