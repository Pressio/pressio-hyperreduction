#!/bin/python

mpirun -np 4 python3 ./test_version.py
mpirun -np 4 python3 ./test_mv.py
mpirun -np 4 python3 ./test_vec.py

mpirun -np 1 python3 ./test_qr_rank1.py
mpirun -np 3 python3 ./test_qr_rank3.py
mpirun -np 1 python3 ./test_svd_rank1.py
mpirun -np 3 python3 ./test_svd_rank3.py

mpirun -np 1 python3 ./test_pseudoinv_rank1.py
mpirun -np 3 python3 ./test_pseudoinv_rank3.py
