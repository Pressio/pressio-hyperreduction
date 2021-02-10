
import pathlib, sys
import numpy as np
import pressiotools as pt

np.set_printoptions(linewidth=140)

def test_version():
  print(pt.__version__)

if __name__ == '__main__':
  test_version()
