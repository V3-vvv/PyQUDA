from os import path
import numpy as np
import cupy as cp
from pyquda import init, enum_quda, pyquda as quda
from pyquda_comm.pointer import ndarrayPointer
from pyquda_utils import core, io, source
from pyquda.field import LatticeInfo, Nc, Ns
from pyquda_plugins import pygwu
from pyquda_utils import convert

init(None, [8, 8, 8, 8], backend="numpy", resource_path=".cache")
latt_info = LatticeInfo([8, 8, 8, 8], 1, 1)
Lx, Ly, Lz, Lt = latt_info.size
kappa = 0.2
rho = 1.5
mass_ori = 0.1
mass = mass_ori / (2 * rho)
num_eigen = 50

evals_kentucky_ov, evecs_kentucky_ov = pygwu.readEigenSystem(
    latt_info, num_eigen, "/home/zhangsh/workspace/tests/gwu_examples/gs8t8iwa165_052.half.overlap.eigensystem", False
)
evals_kentucky_ov = np.array(evals_kentucky_ov, "<c16") / (2 * rho)
evecs_kentucky_ov.data[:, :, :, :, :, :, :2, :] *= -1
ov_eigvals = ndarrayPointer(evals_kentucky_ov)
ov_eigvecs = ndarrayPointer(evecs_kentucky_ov.data_ptrs)

dirac = core.getDirac(latt_info, mass, 1e-8, 100)
dirac.setPrecision(cuda=8, sloppy=8, precondition=8, eigensolver=8)
dirac.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_VERBOSE
dirac.invert_param.mass = mass
dirac.invert_param.kappa = kappa
dirac.invert_param.inv_type = enum_quda.QudaInverterType.QUDA_EIGCG_INVERTER
dirac.invert_param.solution_type = enum_quda.QudaSolutionType.QUDA_MAT_SOLUTION
dirac.invert_param.solve_type = enum_quda.QudaSolveType.QUDA_NORMOP_CHIRAL_SOLVE
dirac.invert_param.dslash_type = enum_quda.QudaDslashType.QUDA_OVERLAP_DSLASH
dirac.invert_param.overlap_invsqrt_tol = 1e-12
dirac.invert_param.ov_n_ev = num_eigen
dirac.invert_param.ov_eigvals = ov_eigvals
dirac.invert_param.ov_eigvecs = ov_eigvecs
# EigenCG param
dirac.invert_param.cuda_prec_ritz = enum_quda.QudaPrecision.QUDA_DOUBLE_PRECISION
dirac.invert_param.n_ev = 50
dirac.invert_param.max_search_dim = dirac.invert_param.n_ev * 3
dirac.invert_param.rhs_idx = 1
dirac.invert_param.deflation_grid = 1
# dirac.invert_param.eigenval_tol = 0.25
# dirac.invert_param.eigcg_max_restarts = 1000
# dirac.invert_param.max_restart_num = 1000
# dirac.invert_param.inc_tol = 1e-8

eig_param_g5w = quda.QudaEigParam()
eig_param_g5w.use_poly_acc = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
eig_param_g5w.poly_deg = 50
eig_param_g5w.a_min = 0.2
eig_param_g5w.a_max = 1 + 8 * kappa
eig_param_g5w.n_ev = 100
eig_param_g5w.n_kr = 300
eig_param_g5w.n_conv = 100
eig_param_g5w.tol = 1e-13
eig_param_g5w.max_restarts = 1000
eig_param_g5w.vec_infile = b"\0"
eig_param_g5w.vec_outfile = b"\0"

gauge = io.readKYUGauge("/home/zhangsh/workspace/tests/gwu_examples/gs8t8iwa165_052", latt_info.global_size)
dirac.loadGauge(gauge)
quda.loadOverlapQuda(dirac.invert_param, eig_param_g5w)

# overlap eigensolver parameters(MdagMChiral)
eig_param_ov = quda.QudaEigParam()
eig_param_ov.invert_param = dirac.invert_param
# eig_param_ov.n_ev = 100
# eig_param_ov.n_kr = 300
# eig_param_ov.n_conv = 100
eig_param_ov.nk = dirac.invert_param.n_ev
eig_param_ov.np = dirac.invert_param.n_ev * dirac.invert_param.deflation_grid
eig_param_ov.tol = 1e-13
eig_param_ov.vec_infile = b"\0"
eig_param_ov.vec_outfile = b"\0"
eig_param_ov.import_vectors = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
eig_param_ov.max_restarts = 100
eig_param_ov.eig_type = enum_quda.QudaEigType.QUDA_EIG_TR_LANCZOS
eig_param_ov.spectrum = enum_quda.QudaEigSpectrumType.QUDA_SPECTRUM_SR_EIG
# dirac type settings
eig_param_ov.use_dagger = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
eig_param_ov.use_norm_op = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
eig_param_ov.use_pc = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
eig_param_ov.compute_gamma5 = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
# eig_param_ov.chirality = enum_quda.QudaChirality.QUDA_CHIRALITY_LEFT
eig_param_ov.location = enum_quda.QudaFieldLocation.QUDA_CUDA_FIELD_LOCATION
eig_param_ov.run_verify = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
# poly_acc settings
eig_param_ov.use_poly_acc = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
eig_param_ov.poly_deg = 10
eig_param_ov.a_min = 0.25
eig_param_ov.a_max = 1

b = source.multiFermion(latt_info, "point", [0, 0, 0, 0])
x = core.MultiLatticeFermion(latt_info, Ns * Nc)
# quda.invertDeflationOverlapQuda(x[0].data_ptr, b[0].data_ptr, eig_param_ov)
for i in range(Ns * Nc):
    if i <= 5:
        eig_param_ov.chirality = enum_quda.QudaChirality.QUDA_CHIRALITY_RIGHT
    else:
        eig_param_ov.chirality = enum_quda.QudaChirality.QUDA_CHIRALITY_LEFT
    quda.invertDeflationOverlapQuda(x[i].data_ptr, b[i].data_ptr, eig_param_ov)

x_ = convert.multiFermionToPropagator(x) / (2 * rho)
x_ref = io.readKYUPropagator(
    f"/home/zhangsh/workspace/tests/gwu_examples/08I_prop_full/gs8t8iwa165_052.grid.000000000_080808008.m{mass_ori:.6f}", latt_info.global_size
)

print((x_ - x_ref).norm2() ** 0.5)
print(np.einsum("wtzyxijab,wtzyxijab->t", x_.data.conj(), x_.data))
print(np.einsum("wtzyxijab,wtzyxijab->t", x_ref.data.conj(), x_ref.data))