
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
def _writeResultsToFile(outDir, projector, isBinary=True, comm=None):
  nullComm = True if comm is None else False
  rank   = 0 if nullComm else comm.Get_rank()

  fileName = outDir + "/galerkinProjector"
  if isBinary:
    fileName+=".bin"
  else:
    fileName+=".txt"

  if nullComm:
    write_array(projector, fileName, isBinary)
  else:
    write_array_distributed(comm, projector, fileName, isBinary)

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
def _computeGalerkinProjectorImpl(phi,
                                  dofsPerNode,
                                  sampleMeshGlobaIndices,
                                  comm,
                                  psi):

  nullComm = True if comm is None else False
  rank     = 0 if nullComm else comm.Get_rank()
  nRanks   = 1 if nullComm else comm.Get_size()

  rowsLocalInds = _getSamplingMatrixLocalInds(sampleMeshGlobaIndices,\
                                              dofsPerNode,\
                                              phi,\
                                              comm)

  if psi==None:
    #########################
    ###    collocation    ###
    #########################
    return None if len(rowsLocalInds)==0 else phi.data()[rowsLocalInds, :]

  else:
    #################
    ###   gappy   ###
    #################
    # compute ((Z*psi)^+)^T psi^T phi
    # note that this is the transpose of the projector operator
    # as defined in the documentation because we want the projector
    # to be row-distribted as the FOM velocity or residual.
    # In general, this projector has MANY rows.

    # phi and psi must have same number of rows
    # but NOT necessarely same num of cols
    assert(phi.data().shape[0] == psi.data().shape[0])

    zpsi = ptla.MultiVector(np.asfortranarray(psi.data()[rowsLocalInds, :]))
    pinvObj = ptla.PseudoInverse()
    pinvObj.compute(zpsi)

    # C = psi^T phi: is replicated on all ranks
    C = np.zeros((psi.extentLocal(1), phi.extentLocal(1)), order='F')
    ptla.product('T','N', 1., psi, phi, 0., C)
    result = pinvObj.applyTranspose(C)
    return None if len(rowsLocalInds)==0 else result


#-----------------------------------------------------------------
def _computeGalerkinProjectorReadYaml(comm, yaml_in):
  # process user inputs
  dic = _processYamlDictionary(yaml_in)

  # check that inputs are valid, exit with error otherwise
  _checkInputsValidity(dic)

  # read state basis matrix
  phi = _readData(dic["dataDir"], dic["StateBasis"], comm)

  # Read sample mesh GLOBAL indices
  smGidFilePath = dic["dataDir"]+"/"+dic["sampMatIndFileName"]
  smGlobalIndices = np.loadtxt(smGidFilePath, dtype=np.int)

  dofsPerNode = dic["dofsPerMnode"]
  hrtype = dic["projMatKind"]
  psi = None if dic["projMatKind"]=="collocation" else _readData(dic["ResidualBasis"], comm)
  projector = _computeGalerkinProjectorImpl(phi, dofsPerNode,
                                            smGlobalIndices,
                                            comm, psi)

  # write projector matrix only if it has data in it
  if isinstance(projector, np.ndarray):
    _writeResultsToFile(dic["outDir"], projector, dic["isBinary"], comm)


#-----------------------------------------------------------------
# this is the entry function visibile outside, all the other functions
# above are implementation details (note the _ prepending all names)
# and should not be exposed outside
#-----------------------------------------------------------------
def computeGalerkinProjector(**args):
  calledFromDriver = True if 'fromDriver' in args.keys() else False

  # if this is called from driverscript
  if calledFromDriver:
    assert(len(args) == 3)
    assert('communicator' in args.keys())
    assert('yamldic'      in args.keys())
    _computeGalerkinProjectorReadYaml(args['communicator'],
                                      args['yamldic'])

  # if not called from driver, it is called from within python
  elif not calledFromDriver:
    assert('stateBasis'         in args.keys())
    assert('sampleMeshIndices'  in args.keys())
    assert('dofsPerMeshNode'    in args.keys())
    comm = None if 'communicator'  not in args.keys() else args['communicator']
    psi  = None if 'residualBasis' not in args.keys() else args['residualBasis']

    return _computeGalerkinProjectorImpl(args["stateBasis"],
                                         args["dofsPerMeshNode"],
                                         args['sampleMeshIndices'],
                                         comm, psi)
