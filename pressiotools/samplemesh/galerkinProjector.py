
import numpy as np
import pathlib, sys
import pressiotools.linalg as ptla
from pressiotools.io.array_read import *
from pressiotools.io.array_write import *
#TODO from pressiotools.sampling_matrix import construct_sampling_matrix


# all the _fnc are implementation details
# check the end for the actual publicly exposed function

#-----------------------------------------------------------------
def _processYamlDictionaryForBasis(yamlDic):
  nCols        = yamlDic["num-columns"]
  dataDir      = yamlDic["dataDir"]
  fileRootName = yamlDic["file-root-name"]
  fileFmt      = yamlDic["format"]

  if fileFmt=="binary":
    isBinary=True
  elif fileFmt=="ascii":
    isBinary=False
  else:
    sys.exit("Unsupported file format: format must be 'binary' or 'ascii'.")

  return {"dataDir"        : dataDir,
          "fileRootName"   : fileRootName,
          "isBinary"       : isBinary,
          "nCols"          : nCols}


#-----------------------------------------------------------------
def _processYamlDictionary(yamlDic):
  dofsPerMnode   = yamlDic["FullMeshInfo"]["dofs-per-mesh-node"]
  
  # state basis info
  stateBasisDic = _processYamlDictionaryForBasis(yamlDic["StateBasis"])

  # residual basis info
  if "ResidualBasis" in yamlDic:
    residBasisDic = _processYamlDictionaryForBasis(yamlDic["ResidualBasis"])
  else:
    residBasisDic = None

  # projector matrix info
  projMatKind        = yamlDic["ProjectorMatrix"]["kind"]
  sampMatIndFileName = yamlDic["ProjectorMatrix"]["sample-mesh-indices-filename"]
  fileFmt            = yamlDic["ProjectorMatrix"]["format"]
  outDir             = yamlDic["ProjectorMatrix"]["outDir"]
  
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
          "outDir"             : outDir}

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
def _createGalerkinProjectorImpl(phi,psi,sampleMeshInds,hyp_type,comm=None):

  # construct sampling matrix
  #TODO sampling_mat, mySMInds = construct_sampling_matrix(sampleMeshInds)
  nSampsLcl = len(sampleMeshInds) #TODO placeholder; replace with the number of local samples in sampling matrix 
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
def _createGalerkinProjectorReadYaml(comm, yaml_in):
  # process user inputs
  dic = _processYamlDictionary(yaml_in)

  # check that inputs are valid, exit with error otherwise
  _checkInputsValidity(dic)

  # read state basis matrix
  phi = _readData(dic["StateBasis"], comm)

  # Read residual basis if required or specified
  if dic["ResidualBasis"]:
    psi = _readData(dic["ResidualBasis"], comm)
  else:
    psi = ptla.MultiVector(phi.data()) # deep copy

  # Read sample mesh indices
  sampleMeshInds = np.loadtxt(dic["sampMatIndFileName"],dtype=np.int)

  # Compute projector matrix
  projector = _createGalerkinProjectorImpl(phi,psi,sampleMeshInds,comm)

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



