#!/bin/python

python test_version.py

# test data structures
mpirun -np 4 python test_vec_mpi.py
mpirun -np 4 python test_mv_mpi.py

# yaml parsing
cd test_yaml_rw
python test_yaml.py
cd ..

# io array
cd test_io
mpirun -np 3 python test_binary_io_mpi.py
mpirun -np 3 python test_ascii_io_mpi.py
cd ..

# lev scores functions
mpirun -np 3 python test_levscores_mpi.py

mpirun -np 3 python test_samplemesh_from_levscores_mpi.py

mpirun -np 1 python test_qr_rank1.py
mpirun -np 3 python test_qr_rank3.py
mpirun -np 1 python test_svd_rank1.py
mpirun -np 3 python test_svd_rank3.py
mpirun -np 1 python test_pseudoinv_rank1.py
mpirun -np 3 python test_pseudoinv_rank3.py
