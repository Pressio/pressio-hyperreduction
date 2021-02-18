
import pathlib, sys
file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(file_path) + "/../../srcpy")

import numpy as np
import pressiotools as pt
from array_io import *

np.set_printoptions(linewidth=140,precision=14)
tol = 1e-14

def run():
  np.random.seed(223)

  # random distributed matrix by selecting
  # rows of a random matrix
  numCols = 4
  mat0 = np.asfortranarray(np.random.rand(15,numCols))
  vec0 = mat0[:,0]

  print(mat0)
  print("---------\n")

  write_array(mat0, "matrix.bin")
  mat0_in = read_array("matrix.bin", numCols)
  assert(np.all(np.abs(mat0_in - mat0) < tol))

  write_array(vec0, "vector.bin")
  vec0_in = read_array("vector.bin", 1)
  assert(np.all(np.abs(vec0_in - vec0) < tol))

if __name__ == '__main__':
  run()
