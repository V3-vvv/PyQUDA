from os import path
import numpy as np
from pyquda import init, enum_quda, pyquda as quda
from pyquda_utils import core, io, source
from pyquda.field import LatticeInfo, Nc, Ns
from pyquda_plugins import pygwu

init(None, [8, 8, 8, 8], backend="numpy", resource_path=".cache")
latt_info = LatticeInfo([8, 8, 8, 8], 1, 1)
Lx, Ly, Lz, Lt = latt_info.size

rho = 1.5
kappa = 0.2
num_eigen = 50

evals_kentucky_ov, evecs_kentucky_ov = pygwu.readEigenSystem(
    latt_info, num_eigen, "/home/suit_zhang/workspace/LatticeQCD/overlap/ensemble/08I/gs8t8iwa165_052.half.overlap.eigensystem", False
)
evals_kentucky_ov = np.array(evals_kentucky_ov, "<c16") / (2 * rho)
evecs_kentucky_ov.data[:, :, :, :, :, :, :2, :] *= -1

num_mass = 3
mass_ori = np.array([0.005, 0.01, 0.02], "<f8")
mass = mass_ori / (2 * rho)
offset = mass**2 / (1 - mass**2)

dirac = core.getDirac(latt_info, mass[0], 1e-8, 100)
dirac.setPrecision(cuda=8, sloppy=8, precondition=8, eigensolver=8)
dirac.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_VERBOSE
dirac.invert_param.mass = mass[0]
dirac.invert_param.kappa = kappa
dirac.invert_param.dslash_type = enum_quda.QudaDslashType.QUDA_OVERLAP_DSLASH
dirac.invert_param.solution_type = enum_quda.QudaSolutionType.QUDA_MAT_SOLUTION
dirac.invert_param.solve_type = enum_quda.QudaSolveType.QUDA_NORMOP_CHIRAL_SOLVE
dirac.invert_param.inv_type = enum_quda.QudaInverterType.QUDA_CG_INVERTER
dirac.invert_param.overlap_invsqrt_tol = 1e-12
dirac.invert_param.num_offset = num_mass
dirac.invert_param.offset = offset.tolist() + [0.0] * (32 - 3)
dirac.invert_param.tol_offset = [1e-8] * 32

eig_param_g5w = quda.QudaEigParam()
eig_param_g5w.use_poly_acc = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
eig_param_g5w.poly_deg = 50
eig_param_g5w.a_min = 0.2
eig_param_g5w.a_max = 1 + 8 * kappa
eig_param_g5w.n_ev = 100
eig_param_g5w.n_kr = 150
eig_param_g5w.n_conv = 100
eig_param_g5w.tol = 1e-13
eig_param_g5w.max_restarts = 1000
eig_param_g5w.vec_infile = b"\0"
eig_param_g5w.vec_outfile = b"\0"

gauge = io.readKYUGauge("/home/suit_zhang/workspace/LatticeQCD/overlap/ensemble/08I/gs8t8iwa165_052", latt_info.global_size)
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

b = source.multiFermion(latt_info, "point", [0, 0, 0, 0])
x = core.MultiLatticeFermion(latt_info, num_mass * Ns * Nc)

b = deflation(evecs_kentucky_ov, b)
for i in range(Ns * Nc):
    if num_mass == 1:
        quda.invertQuda(x[i].data_ptr, b[i].data_ptr, dirac.invert_param)
    else:
        quda.invertMultiShiftQuda(x[num_mass * i : num_mass * (i + 1)].data_ptrs, b[i].data_ptr, dirac.invert_param)
