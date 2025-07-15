from os import path
import numpy as np
from pyquda import init, enum_quda, pyquda as quda
from pyquda_comm.pointer import ndarrayPointer
from pyquda_utils import core, io, source
from pyquda.field import LatticeInfo, LatticePropagator, MultiLatticeFermion, Nc, Ns
from pyquda_plugins import pygwu

# init([1, 1, 1, 1], [8, 8, 8, 8], backend="numpy", resource_path=".cache")
# latt_info = LatticeInfo([8, 8, 8, 8], 1, 1)
init([1, 1, 1, 4], [24, 24, 24, 64], backend="numpy", resource_path=".cache")
latt_info = LatticeInfo([24, 24, 24, 64], 1, 1)
Lx, Ly, Lz, Lt = latt_info.size
kappa = 0.2
rho = 1.5
# mass_ori = 0.005
mass_ori = 0.6
mass = mass_ori / (2 * rho)
# num_eigen = 50
num_eigen = 100

# evals_kentucky_ov, evecs_kentucky_ov = pygwu.readEigenSystem(
#     latt_info, num_eigen, "/home/zhangsh/workspace/tests/gwu_examples/gs8t8iwa165_052.half.overlap.eigensystem", False
# )
evals_kentucky_ov, evecs_kentucky_ov = pygwu.readEigenSystem(
    latt_info, num_eigen, "/public/ensemble/24I/rbc_conf_2464_m0.005_0.04_008545_hyp.half.overlap.eigensystem", False
)
evals_kentucky_ov = np.array(evals_kentucky_ov, "<c16") / (2 * rho)
evecs_kentucky_ov.data[:, :, :, :, :, :, :2, :] *= -1
print(f"evecs_kentucky_ov shape: {evecs_kentucky_ov.data.shape}")

dirac = core.getDirac(latt_info, mass, 1e-8, 1000)
dirac.setPrecision(cuda=8, sloppy=8, precondition=2, eigensolver=8)
dirac.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_VERBOSE
dirac.invert_param.mass = mass
dirac.invert_param.kappa = kappa
dirac.invert_param.dslash_type = enum_quda.QudaDslashType.QUDA_OVERLAP_DSLASH
dirac.invert_param.solution_type = enum_quda.QudaSolutionType.QUDA_MAT_SOLUTION
dirac.invert_param.solve_type = enum_quda.QudaSolveType.QUDA_NORMOP_CHIRAL_SOLVE
dirac.invert_param.inv_type = enum_quda.QudaInverterType.QUDA_CG_INVERTER
dirac.invert_param.overlap_invsqrt_tol = 1e-12
# dirac.invert_param.ov_n_ev = num_eigen
# ov_eigvals = ndarrayPointer(evals_kentucky_ov)
# ov_eigvecs = ndarrayPointer(evecs_kentucky_ov.data_ptrs)
# dirac.invert_param.ov_eigvals = ov_eigvals
# dirac.invert_param.ov_eigvecs = ov_eigvecs

eig_param_g5w = quda.QudaEigParam()
eig_param_g5w.use_poly_acc = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
eig_param_g5w.poly_deg = 100
eig_param_g5w.a_min = 0.3
eig_param_g5w.a_max = 1 + 8 * kappa
eig_param_g5w.n_ev = 150
eig_param_g5w.n_kr = 200
eig_param_g5w.n_conv = 150
eig_param_g5w.tol = 1e-13
eig_param_g5w.max_restarts = 1000
eig_param_g5w.vec_infile = b"\0"
eig_param_g5w.vec_outfile = b"\0"

# gauge = io.readKYUGauge("/home/zhangsh/workspace/tests/gwu_examples/gs8t8iwa165_052", latt_info.global_size)
gauge = io.readKYUGauge("/public/ensemble/24I/rbc_conf_2464_m0.005_0.04_008545_hyp", latt_info.global_size)
dirac.loadGauge(gauge)
quda.loadOverlapQuda(dirac.invert_param, eig_param_g5w)

def apply_gamma5(vec):
    # gamma5 = diag(-1,-1,1,1)
    g5 = np.array([-1, -1, 1, 1], dtype=vec.data.dtype)
    return vec.data * g5[None, None, None, None, None, :, None]

def deflation(evecs_kentucky_ov, b):
    """
    对 b 进行 deflation，返回 b_defl。
    evecs_kentucky_ov: 本征矢对象
    b: MultiLatticeFermion
    返回: b_defl, 与 b 结构相同
    """
    num_eigen = evecs_kentucky_ov.data.shape[0]
    b_defl = b.copy()
    for i in range(b.L5):
        # 拷贝b[i]到b_defl[i]
        b_defl[i].data[:] = b[i].data
        for j in range(num_eigen):
            # 正交投影
            alpha = np.vdot(evecs_kentucky_ov.data[j].ravel(), b_defl[i].data.ravel())
            b_defl[i].data -= alpha * evecs_kentucky_ov.data[j]
            # gamma5投影
            gamma5_evec = apply_gamma5(evecs_kentucky_ov[j])
            alpha_g5 = np.vdot(gamma5_evec.ravel(), b_defl[i].data.ravel())
            b_defl[i].data -= alpha_g5 * gamma5_evec
    return b_defl

# 只计算高模部分，且对b做deflation
mrhs = 6
restart = 0
propagator = LatticePropagator(latt_info)
for s in range(0, Ns * Nc, mrhs):
    b = MultiLatticeFermion(latt_info, min(mrhs, Ns * Nc - s))
    for i in range(b.L5):
        b[i] = source.source(latt_info, "point", [0, 0, 0, 0], (s + i) // Nc, (s + i) % Nc, None)
    b_defl = deflation(evecs_kentucky_ov, b)
    x = dirac.invertMultiSrcRestart(b_defl, restart)
    for i in range(b.L5):
        propagator.setFermion(x[i], (s + i) // Nc, (s + i) % Nc)
        
# b = source.multiFermion(latt_info, "point", [0, 0, 0, 0])
# x = core.MultiLatticeFermion(latt_info, Ns * Nc)
# b_defl = deflation(evecs_kentucky_ov, b)
# for i in range(Ns * Nc):
#     quda.invertQuda(x[i].data_ptr, b_defl[i].data_ptr, dirac.invert_param)

# print(f"propagator shape: {propagator.data.shape}")
# propagator_ = propagator / (2 * rho)
# propagator_ref = io.readKYUPropagator(
#     f"/home/zhangsh/workspace/tests/gwu_examples/08I_prop_high/multi_shift/gs8t8iwa165_052.grid.000000000_080808008.m{mass_ori:.6f}", latt_info.global_size
# )
# propagator_ref = io.readKYUPropagator(
#     f"/public/ensemble/24I/rbc_conf_2464_m0.005_0.04_008545_hyp.grid.000000000_242424064.m{mass_ori:.6f}", latt_info.global_size
# )
# print((propagator_ - propagator_ref).norm2() ** 0.5)
# print(np.einsum("wtzyxijab,wtzyxijab->t", propagator_.data.conj(), propagator_.data))
# print(np.einsum("wtzyxijab,wtzyxijab->t", propagator_ref.data.conj(), propagator_ref.data))