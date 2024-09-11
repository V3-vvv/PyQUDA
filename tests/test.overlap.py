import os
import sys

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
sys.path.insert(1, ".")

import numpy as np
import cupy as cp
from check_pyquda import weak_field
from pyquda import init, core, quda, enum_quda
from pyquda.dirac import setPrecision
from pyquda.pointer import ndarrayDataPointer
from pyquda.field import LatticeFermion, LatticeInfo, Nc, Ns
from pyquda.utils import source, io

init([1, 1, 1, 1])
latt_info = LatticeInfo([8, 8, 8, 8], 1, 1)
Lx, Ly, Lz, Lt = latt_info.size

# hwilson_eigenvector = np.fromfile(
#     "/mnt/datadisk0/jinzhi/zhangsh/test/gwu_examples/gs8t8iwa165_052.Hwilson.eigensystem", "<c16"
# ).reshape(100, 8, 8, 8, 8, 4, 3)

gauge = io.readKYUGauge("/mnt/datadisk0/jinzhi/zhangsh/test/gwu_examples/gs8t8iwa165_052", [8, 8, 8, 8])

kappa = 0.2
mass = 1 / (2 * kappa) - 4

setPrecision(cuda=8, sloppy=8, precondition=8, eigensolver=8)
dirac = core.getDirac(latt_info, mass, 2e-15, 1000)
dirac.loadGauge(gauge)
dirac.invert_param.mass_normalization = enum_quda.QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
dirac.invert_param.dagger = enum_quda.QudaDagType.QUDA_DAG_YES

eig_param = quda.QudaEigParam()
eig_param.invert_param = dirac.invert_param
eig_param.eig_type = enum_quda.QudaEigType.QUDA_EIG_TR_LANCZOS
eig_param.use_poly_acc = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
eig_param.poly_deg = 50
eig_param.a_min = 0.2**2
eig_param.a_max = (1 + 8 * 0.2) ** 2
eig_param.use_dagger = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
eig_param.use_norm_op = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
eig_param.use_pc = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
eig_param.compute_gamma5 = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
# Only LR or LM can be used if use_poly_acc = QUDA_BOOLEAN_TRUE
# The actual spectrum is determined by a_min and a_max
eig_param.spectrum = enum_quda.QudaEigSpectrumType.QUDA_SPECTRUM_LR_EIG
eig_param.n_ev = 100
eig_param.n_kr = 300
eig_param.n_conv = 100
eig_param.tol = 1e-13
eig_param.vec_infile = b""
eig_param.vec_outfile = b""
eig_param.max_restarts = 1000

evecs = cp.zeros((200, 2, Lt, Lz, Ly, Lx, Ns, Nc), "<c16")
evals = np.zeros((100), "<c16")
quda.eigensolveQuda(ndarrayDataPointer(evecs.reshape(200, -1), True), evals, eig_param)
print(evals)

dirac.invert_param.dslash_type = enum_quda.QudaDslashType.QUDA_OVERLAP_DSLASH
dirac.invert_param.solve_type = enum_quda.QudaSolveType.QUDA_DIRECT_SOLVE
dirac.invert_param.inv_type = enum_quda.QudaInverterType.QUDA_BICGSTAB_INVERTER
dirac.invert_param.hermitian_wilson_n_ev = 100
dirac.invert_param.hermitian_wilson_n_kr = 300
dirac.invert_param.hermitian_wilson_tol = 1e-13
dirac.invert_param.overlap_invsqrt_tol = 1e-12

x = source.source(latt_info, "point", [0, 0, 0, 0], 0, 0)
for spin in range(4):
    for color in range(3):
        x.data[0, 0, 0, 0, 0, spin, color] = 1
b = LatticeFermion(latt_info)

quda.MatQuda(b.data_ptr, x.data_ptr, dirac.invert_param)

print(x.data[0, 0, 0, 0, 0])
print(b.data[0, 0, 0, 0, 0])