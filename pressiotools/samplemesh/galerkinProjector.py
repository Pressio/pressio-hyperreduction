
import numpy as np
import pathlib, sys
import pressiotools.linalg as ptla
from pressiotools.io.array_read import *
from pressiotools.io.array_write import *
#TODO from pressiotools.sampling_matrix import construct_sampling_matrix

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# all the _fnc are implementation details
# check the end for the actual publicly exposed function
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def _processYamlDictionaryForBasis(yamlDic):
  nCols        = yamlDic["num-columns"]
  fileRootName = yamlDic["file-root-name"]
  fileFmt      = yamlDic["format"]
  if fileFmt=="binary":
    isBinary=True
  elif fileFmt=="ascii":
    isBinary=False
  else:
    sys.exit("Unsupported file format: format must be 'binary' or 'ascii'.")

  return {"fileRootName"   : fileRootName,
          "isBinary"       : isBinary,
          "nCols"          : nCols}

#-----------------------------------------------------------------
def _processYamlDictionary(yamlDic):
  # dataDir is not an entry specified in the real yaml file
  # so you won't find it inside the inputs-templates/galerkin-projector-template.yaml
  # because it is passed to driver script as cmd line arg
  # and inserted in the dic by the driver script
  dataDir      = yamlDic["dataDir"]

  dofsPerMnode = yamlDic["FullMeshInfo"]["dofs-per-mesh-node"]

  # state basis info
  stateBasisDic = _processYamlDictionaryForBasis(yamlDic["StateBasis"])

  # residual basis info
  if "ResidualBasis" in yamlDic:
    residBasisDic = _processYamlDictionaryForBasis(yamlDic["ResidualBasis"])
  else:
    residBasisDic = None

  # projector matrix info
  outDir             = yamlDic["ProjectorMatrix"]["outDir"]
  projMatKind        = yamlDic["ProjectorMatrix"]["kind"]
  sampMatIndFileName = yamlDic["ProjectorMatrix"]["sample-mesh-indices-filename"]
  fileFmt            = yamlDic["ProjectorMatrix"]["format"]
  if fileFmt=="binary":
    isBinary=True
  elif fileFmt=="ascii":
    isBinary=False
  else:
    sys.exit("Unsupported file format: format must be 'binary' or 'ascii'.")

  return {"dofsPerMnode"       : dofsPerMnode,
          "StateBasis"         : stateBasisDic,
          "ResidualBasis"      : residBasisDic,
          "projMatKind"        : projMatKind,
          "sampMatIndFileName" : sampMatIndFileName,
          "isBinary"           : isBinary,
          "outDir"             : outDir,
          "dataDir"            : dataDir}

#-----------------------------------------------------------------
def _checkInputsValidity(dic):
  if dic["dofsPerMnode"] < 1:
    sys.exit("unsupported input: ResidualBasis/dofs-per-mesh-node needs to be greater than 0")

  if dic["StateBasis"]["nCols"] < 1:
    sys.exit("unsupported input: StateBasis/nCols needs to be greater than 0")

  if dic["ResidualBasis"] and dic["ResidualBasis"]["nCols"] < 1:
    sys.exit("unsupported input: ResidualBasis/nCols needs to be greater than 0")

  if (dic["projMatKind"] is "gappy-pod") and (dic["ResidualBasis"] is None):
    sys.exit("Gappy POD requires a residual basis")

#-----------------------------------------------------------------
def _readData(dataDir, dic, comm=None):
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
def _getSamplingMatrixLocalInds(sampleMeshGlobalInds, dofsPerMnode, matrix, comm=None):
  # assume that global indices are:
  # i_rank * n_node_per_rank + i_array_local, i_rank < n_ranks

  nullComm = True if comm is None else False
  rank   = 0 if nullComm else comm.Get_rank()
  nRanks = 1 if nullComm else comm.Get_size()

  myNumRows = matrix.extentLocal(0)
  myMinRowGid = matrix.minRowGidLocal()
  myMaxRowGid = matrix.maxRowGidLocal()

  # determine my global mesh indices
  myNumMeshNodes   = int(myNumRows/dofsPerMnode)
  myMinMeshGid     = int(myMinRowGid/dofsPerMnode)
  myGlobalMeshInds = myMinMeshGid + np.arange(myNumMeshNodes)

  if myMinMeshGid < 0: sys.exit("Global Indices must be positive")
 
  sampleMeshLocalInds = np.intersect1d(sampleMeshGlobalInds, myGlobalMeshInds) - myMinMeshGid
  
  print(rank,sampleMeshLocalInds + myMinMeshGid)
  
  # convert to indices for state vector or basis matrix 
  sampleMatrixLocalInds = np.repeat(sampleMeshLocalInds, dofsPerMnode) \
                          + np.repeat(np.arange(dofsPerMnode), len(sampleMeshLocalInds))
  return sampleMatrixLocalInds

#-----------------------------------------------------------------
def _writeResultsToFile(outDir, projector, isBinary=True, comm=None):
  nullComm = True if comm is None else False
  rank   = 0 if nullComm else comm.Get_rank()

  fileName = outDir + "/galerkinProjector"
  if isBinary:
    fileName+=".bin"
  else:
    fileName+=".txt"

  if nullComm:
    write_array(projector.data(), fileName, isBinary)
  else:
    write_array_distributed(comm, projector, fileName, isBinary)

#-----------------------------------------------------------------
def _createGappyProjectorImpl(phi, psi, smGlobalIndices, comm=None):
  # construct sampling matrix
  #TODO sampling_mat, mySMInds = construct_sampling_matrix(smGlobalIndices)
  nSampsLcl = len(smGlobalIndices) #TODO placeholder; replace with the number of local samples in sampling matrix
  nCols = psi.data().shape[1]
  projector = ptla.MultiVector(np.asfortranarray(np.random.rand(nCols,nSampsLcl)))
  #if hyp_type == "collocation":
  #TODO ptla.product(transpose, transpose, 1., state_basis, sampling_mat, 0, projector)
  #elif hyp_type =="gappy_pod":
  #TODO B = sampling_mat * res_basis
  #TODO B1 = ptla.lingalg.pseudo_inverse(B)
  #TODO ptla.product(transpose, nontranspose, 1., state_basis, B1, 0, projector)
  return projector

#-----------------------------------------------------------------
def _createCollocationProjectorImpl(phi, samplingMatrixLocalInds):
  return ptla.MultiVector(np.asfortranarray(phi.data()[samplingMatrixLocalInds, :]))

#-----------------------------------------------------------------
def _createGalerkinProjectorReadYaml(comm, yaml_in):
  # process user inputs
  dic = _processYamlDictionary(yaml_in)

  # check that inputs are valid, exit with error otherwise
  _checkInputsValidity(dic)

  # read state basis matrix
  phi = _readData(dic["dataDir"], dic["StateBasis"], comm)

  # Read sample mesh GLOBAL indices
  smGlobInds = np.loadtxt(dic["dataDir"]+"/"+dic["sampMatIndFileName"], dtype=np.int)

  # Determine vector global indices from mesh indices
  samplingMatrixLocalInds = _getSamplingMatrixLocalInds(smGlobInds,\
                                                        dic["dofsPerMnode"],\
                                                        phi,\
                                                        comm)

  # dispatch computation
  hrtype = dic["projMatKind"]
  if hrtype == "gappy-pod":
    # we need the state and residual basis for this
    psi = _readData(dic["ResidualBasis"], comm)
    #projector = _createGappyProjectorImpl(phi, psi, smGlobInds, comm)
    sys.exit("Gal proj for gappy NOT implemented yet")

  elif hrtype == "collocation":
    # we only reed the state basis for this
    projector = _createCollocationProjectorImpl(phi, samplingMatrixLocalInds)

  else:
    sys.exit("Invalid hyper-reduction selection")

  if projector == None:
    sys.exit("Computed invalid projector, something went wrong")

  # write projector matrix to disk
  _writeResultsToFile(dic["outDir"], projector, dic["isBinary"], comm)

#-----------------------------------------------------------------
# this is the entry function visibile outside, all the other functions
# above are implementation details (note the _ prepending all names)
# and should not be exposed outside
#-----------------------------------------------------------------
def createGalerkinProjector(**args):
  if len(args) == 2 \
     and 'communicator' in args.keys() \
     and 'yamldic'      in args.keys():
    _createGalerkinProjectorReadYaml(args['communicator'],
                                     args['yamldic'])
