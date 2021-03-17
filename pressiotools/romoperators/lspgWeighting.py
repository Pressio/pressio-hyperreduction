
import numpy as np
import pathlib, sys
import pressiotools.linalg as ptla
from pressiotools.io.array_read import *
from pressiotools.io.array_write import *

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
  # so you won't find it inside the inputs-templates/...template.yaml
  # because it is passed to driver script as cmd line arg
  # and inserted in the dic by the driver script
  dataDir      = yamlDic["dataDir"]

  dofsPerMnode = yamlDic["FullMeshInfo"]["dofs-per-mesh-node"]

  # residual basis info
  residBasisDic = _processYamlDictionaryForBasis(yamlDic["ResidualBasis"])

  # weighting matrix info
  outDir             = yamlDic["WeightingMatrix"]["outDir"]
  weightMatKind      = yamlDic["WeightingMatrix"]["kind"]
  sampMatIndFileName = yamlDic["WeightingMatrix"]["sample-mesh-indices-filename"]
  fileFmt            = yamlDic["WeightingMatrix"]["format"]
  if fileFmt=="binary":
    isBinary=True
  elif fileFmt=="ascii":
    isBinary=False
  else:
    sys.exit("Unsupported file format: format must be 'binary' or 'ascii'.")

  return {"dofsPerMnode"       : dofsPerMnode,
          "ResidualBasis"      : residBasisDic,
          "weightMatKind"        : weightMatKind,
          "sampMatIndFileName" : sampMatIndFileName,
          "isBinary"           : isBinary,
          "outDir"             : outDir,
          "dataDir"            : dataDir}

#-----------------------------------------------------------------
def _checkInputsValidity(dic):
  if dic["dofsPerMnode"] < 1:
    sys.exit("unsupported input: ResidualBasis/dofs-per-mesh-node needs to be greater than 0")

  if dic["ResidualBasis"] and dic["ResidualBasis"]["nCols"] < 1:
    sys.exit("unsupported input: ResidualBasis/nCols needs to be greater than 0")

  if dic["weightMatKind"] != "gnat":
    sys.exit("In yaml file, you set: *{}*, but LSPG Weighting only supports GNAT".format(dic["weightMatKind"]))

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
def _writeResultsToFile(outDir, weighting, isBinary=True, comm=None):
  nullComm = True if comm is None else False
  rank   = 0 if nullComm else comm.Get_rank()

  fileName = outDir + "/lspgWeighting"
  if isBinary:
    fileName+=".bin"
  else:
    fileName+=".txt"

  if nullComm:
    write_array(weighting, fileName, isBinary)
  else:
    write_array_distributed(comm, weighting, fileName, isBinary)

#-----------------------------------------------------------------
def _getSamplingMatrixLocalInds(sampleMeshGlobalInds, dofsPerMnode, matrix, comm=None):
  # assume that global indices are:
  # i_rank * n_node_per_rank + i_array_local, i_rank < n_ranks

  nullComm = True if comm is None else False
  rank   = 0 if nullComm else comm.Get_rank()
  nRanks = 1 if nullComm else comm.Get_size()

  if len(sampleMeshGlobalInds) == 0:
    return []
  else:
    myNumRows = matrix.extentLocal(0)
    myMinRowGid = matrix.minRowGidLocal()
    myMaxRowGid = matrix.maxRowGidLocal()

    # determine my global mesh indices
    myNumMeshNodes   = int(myNumRows/dofsPerMnode)
    myMinMeshGid     = int(myMinRowGid/dofsPerMnode)
    myGlobalMeshInds = myMinMeshGid + np.arange(myNumMeshNodes)

    if myMinMeshGid < 0: sys.exit("Global Indices must be positive")

    sampleMeshLocalInds = np.intersect1d(sampleMeshGlobalInds, myGlobalMeshInds) - myMinMeshGid

    torep = sampleMeshLocalInds*dofsPerMnode+myMinRowGid
    shift = np.tile(np.arange(dofsPerMnode), len(sampleMeshLocalInds))
    rowsLocalInds = np.repeat(torep, dofsPerMnode) + shift - myMinRowGid
    return rowsLocalInds.tolist()

#-----------------------------------------------------------------
def _computeLspgWeightingImpl(psi,
                              dofsPerNode,
                              sampleMeshGlobaIndices,
                              comm):

  nullComm = True if comm is None else False
  rank     = 0 if nullComm else comm.Get_rank()
  nRanks   = 1 if nullComm else comm.Get_size()

  rowsLocalInds = _getSamplingMatrixLocalInds(sampleMeshGlobaIndices,\
                                              dofsPerNode,\
                                              psi,\
                                              comm)
  # here we compute:
  # ((Z*psi)^+)^T psi^T psi (Z*psi)^+

  # compute (Z*psi)^+
  zpsi = ptla.MultiVector(np.asfortranarray(psi.data()[rowsLocalInds, :]))
  pinvObj = ptla.PseudoInverse()
  pinvObj.compute(zpsi)

  # compute B = psi^T psi, where B is replicated on each rank
  B = np.zeros((psi.extentLocal(1), psi.extentLocal(1)), order='F')
  ptla.selfTransposeSelf(1., psi, B)

  # compute C = ((Z*psi)^+)^T psi^T psi
  C = ptla.MultiVector(pinvObj.applyTranspose(B))

  # compute target result
  zpsiT = ptla.MultiVector(pinvObj.viewTransposeLocal())
  D = np.zeros((C.extentLocal(0), zpsiT.extentGlobal(0)), order='F')
  ptla.product('N','T', 1., C, zpsiT, 0., D)

  return None if len(rowsLocalInds)==0 else D

#-----------------------------------------------------------------
def _computeLspgWeightingReadYaml(comm, yaml_in):
  # process user inputs
  dic = _processYamlDictionary(yaml_in)

  # check that inputs are valid, exit with error otherwise
  _checkInputsValidity(dic)

  # read residual basis matrix
  psi = _readData(dic["dataDir"], dic["ResidualBasis"], comm)

  # Read sample mesh GLOBAL indices
  smGidFilePath = dic["dataDir"]+"/"+dic["sampMatIndFileName"]
  smGlobalIndices = np.loadtxt(smGidFilePath, dtype=np.int)

  dofsPerNode = dic["dofsPerMnode"]
  weighting = _computeLspgWeightingImpl(psi, dofsPerNode,smGlobalIndices,comm)

  # write weighting matrix only if it has data in it
  if isinstance(weighting, np.ndarray):
    _writeResultsToFile(dic["outDir"], weighting, dic["isBinary"], comm)

#---------------------------------------------------------------------
# this is the entry function visibile outside, all the other functions
# above are implementation details (note the _ prepending all names)
# and should not be exposed outside
#---------------------------------------------------------------------
def computeLspgWeighting(**args):
  calledFromDriver = True if 'fromDriver' in args.keys() else False

  # if this is called from driverscript
  if calledFromDriver:
    assert(len(args) == 3)
    assert('communicator' in args.keys())
    assert('yamldic'      in args.keys())
    _computeLspgWeightingReadYaml(args['communicator'],
                                  args['yamldic'])

  # if not called from driver, it is called from within python
  elif not calledFromDriver:
    assert('residualBasis'      in args.keys())
    assert('sampleMeshIndices'  in args.keys())
    assert('dofsPerMeshNode'    in args.keys())
    comm = None if 'communicator'  not in args.keys() else args['communicator']

    return _computeLspgWeightingImpl(args["residualBasis"],
                                     args["dofsPerMeshNode"],
                                     args['sampleMeshIndices'],
                                     comm)
