#!/bin/python

mpirun -np 4 python ./test_version.py

mpirun -np 4 python ./test_mv.py
mpirun -np 4 python ./test_vec.py

mpirun -np 1 python ./test_qr_rank1.py
mpirun -np 3 python ./test_qr_rank3.py
mpirun -np 1 python ./test_svd_rank1.py
mpirun -np 3 python ./test_svd_rank3.py
mpirun -np 1 python ./test_pseudoinv_rank1.py
mpirun -np 3 python ./test_pseudoinv_rank3.py

mpirun -np 3 python ./test_levscores.py

cd test_io
mpirun -np 3 python ./test_binary_io.py
mpirun -np 3 python ./test_ascii_io.py
cd ..

cd test_yaml_rw
python ./test_yaml.py
cd ..
