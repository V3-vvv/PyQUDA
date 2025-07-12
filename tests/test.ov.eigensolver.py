import sys
import numpy as np
import cupy as cp
import check_pyquda
from pyquda import init, enum_quda, pyquda as quda
from pyquda_utils import core, io
from pyquda.dirac import setPrecision
from pyquda.field import LatticeInfo, Nc, Ns

init(None, [8, 8, 8, 8], backend="numpy", resource_path=".cache")
latt_info = LatticeInfo([8, 8, 8, 8], 1, 1)
Lx, Ly, Lz, Lt = latt_info.size
kappa = 0.2
mass = 1 / (2 * kappa) - 4

dirac = core.getDirac(latt_info, mass, 2e-12, 1000)
gauge = io.readKYUGauge("/home/zhangsh/workspace/tests/gwu_examples/gs8t8iwa165_052", [8, 8, 8, 8])
setPrecision(cuda=8, sloppy=8, precondition=8, eigensolver=8)
dirac.loadGauge(gauge)
dirac.invert_param.mass = 0
dirac.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_VERBOSE
dirac.invert_param.mass_normalization = enum_quda.QudaMassNormalization.QUDA_KAPPA_NORMALIZATION
dirac.invert_param.dslash_type = enum_quda.QudaDslashType.QUDA_OVERLAP_DSLASH
dirac.invert_param.overlap_invsqrt_tol = 1e-12

# H_wilson eigensolver parameters
hw_eig_param = quda.QudaEigParam()
hw_eig_param.n_ev = 100
hw_eig_param.n_kr = 300
hw_eig_param.n_conv = 100
hw_eig_param.tol = 1e-13
hw_eig_param.max_restarts = 100
hw_eig_param.vec_infile = b"\0"
hw_eig_param.vec_outfile = b"\0"
# poly_acc settings
hw_eig_param.use_poly_acc = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
hw_eig_param.poly_deg = 100
hw_eig_param.a_min = 0.18
hw_eig_param.a_max = (1 + 8 * kappa)

quda.loadOverlapQuda(dirac.invert_param, hw_eig_param)

# # overlap eigensolver parameters(M)
# ov_eig_param = quda.QudaEigParam()
# ov_eig_param.invert_param = dirac.invert_param
# ov_eig_param.n_ev = 100
# ov_eig_param.n_kr = 300
# ov_eig_param.n_conv = 100
# ov_eig_param.tol = 1e-13
# ov_eig_param.vec_infile = b"\0"
# ov_eig_param.vec_outfile = b"\0"
# ov_eig_param.max_restarts = 100
# ov_eig_param.eig_type = enum_quda.QudaEigType.QUDA_EIG_IR_ARNOLDI
# ov_eig_param.spectrum = enum_quda.QudaEigSpectrumType.QUDA_SPECTRUM_SM_EIG
# # dirac type settings
# ov_eig_param.use_dagger = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
# ov_eig_param.use_norm_op = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
# ov_eig_param.use_pc = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
# ov_eig_param.compute_gamma5 = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
# ov_eig_param.chirality = enum_quda.QudaChirality.QUDA_CHIRALITY_INVALID

# # overlap eigensolver parameters(MdagM)
# ov_eig_param = quda.QudaEigParam()
# ov_eig_param.invert_param = dirac.invert_param
# ov_eig_param.n_ev = 100
# ov_eig_param.n_kr = 300
# ov_eig_param.n_conv = 100
# ov_eig_param.tol = 1e-13
# ov_eig_param.vec_infile = b"\0"
# ov_eig_param.vec_outfile = b"\0"
# ov_eig_param.max_restarts = 100
# ov_eig_param.eig_type = enum_quda.QudaEigType.QUDA_EIG_TR_LANCZOS
# ov_eig_param.spectrum = enum_quda.QudaEigSpectrumType.QUDA_SPECTRUM_SR_EIG
# # dirac type settings
# ov_eig_param.use_dagger = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
# ov_eig_param.use_norm_op = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
# ov_eig_param.use_pc = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
# ov_eig_param.compute_gamma5 = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
# ov_eig_param.chirality = enum_quda.QudaChirality.QUDA_CHIRALITY_INVALID
# # poly_acc settings
# ov_eig_param.use_poly_acc = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
# ov_eig_param.poly_deg = 10
# ov_eig_param.a_min = 0.15
# ov_eig_param.a_max = 1

# ov_evecs = np.zeros((ov_eig_param.n_ev, 2, Lt, Lz, Ly, Lx//2, Ns, Nc), "<c16")
# ov_evals = np.zeros((ov_eig_param.n_ev), "<c16")
# quda.eigensolveQuda(ov_evecs.reshape(ov_eig_param.n_ev, -1), ov_evals, ov_eig_param)

# overlap eigensolver parameters(MdagMChiral)
ov_eig_param = quda.QudaEigParam()
ov_eig_param.invert_param = dirac.invert_param
ov_eig_param.n_ev = 100
ov_eig_param.n_kr = 300
ov_eig_param.n_conv = 100
ov_eig_param.tol = 1e-13
ov_eig_param.vec_infile = b"\0"
ov_eig_param.vec_outfile = b"\0"
ov_eig_param.max_restarts = 100
ov_eig_param.eig_type = enum_quda.QudaEigType.QUDA_EIG_TR_LANCZOS
ov_eig_param.spectrum = enum_quda.QudaEigSpectrumType.QUDA_SPECTRUM_SR_EIG
# dirac type settings
ov_eig_param.use_dagger = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
ov_eig_param.use_norm_op = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
ov_eig_param.use_pc = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
ov_eig_param.compute_gamma5 = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
ov_eig_param.chirality = enum_quda.QudaChirality.QUDA_CHIRALITY_LEFT
# poly_acc settings
ov_eig_param.use_poly_acc = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
ov_eig_param.poly_deg = 10
ov_eig_param.a_min = 0.25
ov_eig_param.a_max = 1

ov_evecs = np.zeros((ov_eig_param.n_ev, 2, Lt, Lz, Ly, Lx//2, Ns//2, Nc), "<c16")
ov_evals = np.zeros((ov_eig_param.n_ev), "<c16")
quda.eigensolveQuda(ov_evecs.reshape(ov_eig_param.n_ev, -1), ov_evals, ov_eig_param)

sys.exit(0)

np.save("evals_ov.npy", ov_evals)
np.save("evecs_ov.npy", ov_evecs)