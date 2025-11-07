from os import path
from check_pyquda import weak_field
from pyquda import init
from pyquda_utils import io
from pyquda.field import LatticeInfo

import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

init(None, [144, 144, 144, 288], backend="numpy", resource_path=".cache")
latt_info = LatticeInfo([144, 144, 144, 288], 1, 1)

start = time.perf_counter()
gauge = io.readKYUGauge("/public/home/ybyang_1/data/a045m130/l144288f211b700m000569m01555m1827a.scidac.1014.01_hyp", latt_info.global_size)
elapsed = time.perf_counter() - start
if rank == 1:
    print(f"====================readKYUGauge elapsed: {elapsed:.6f} s.====================")

gauge.smearHYP(0, 0.75, 0.6, 0.3, 4, compute_plaquette=True)
