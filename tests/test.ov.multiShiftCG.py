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
num_eigen = 50

evals_kentucky_ov, evecs_kentucky_ov = pygwu.readEigenSystem(
    latt_info, num_eigen, "/home/zhangsh/workspace/tests/gwu_examples/gs8t8iwa165_052.half.overlap.eigensystem", False
)
evals_kentucky_ov = np.array(evals_kentucky_ov, "<c16") / (2 * rho)
evecs_kentucky_ov.data[:, :, :, :, :, :, :2, :] *= -1

print(evals_kentucky_ov)
print(evecs_kentucky_ov.data[1, 0, 0, 0, 0, 0, :, :])
# mass_ori = np.array([0.1, 0.2, 0.5], "<f8")
# mass_ori = np.array([0.005, 0.01, 0.02], "<f8")
mass_ori = np.array([0.2, 0.3, 0.4], "<f8")
mass = mass_ori / (2 * rho)
offset = mass**2 / (1 - mass**2)
num_mass = 3

dirac = core.getDirac(latt_info, mass[0], 1e-8, 100)
dirac.setPrecision(cuda=8, sloppy=8, precondition=2, eigensolver=8)
dirac.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_VERBOSE
dirac.invert_param.mass = mass[0]
dirac.invert_param.kappa = kappa
dirac.invert_param.dslash_type = enum_quda.QudaDslashType.QUDA_OVERLAP_DSLASH
dirac.invert_param.solution_type = enum_quda.QudaSolutionType.QUDA_MAT_SOLUTION
dirac.invert_param.solve_type = enum_quda.QudaSolveType.QUDA_NORMOP_CHIRAL_SOLVE
dirac.invert_param.inv_type = enum_quda.QudaInverterType.QUDA_CG_INVERTER
dirac.invert_param.overlap_invsqrt_tol = 1e-12
dirac.invert_param.ov_n_ev = num_eigen
dirac.invert_param.num_offset = num_mass
dirac.invert_param.offset = offset.tolist() + [0.0] * (32 - 3)
dirac.invert_param.tol_offset = [1e-8] * 32
ov_eigvals = ndarrayPointer(evals_kentucky_ov)
ov_eigvecs = ndarrayPointer(evecs_kentucky_ov.data_ptrs)
dirac.invert_param.ov_eigvals = ov_eigvals
dirac.invert_param.ov_eigvecs = ov_eigvecs

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

b = source.multiFermion(latt_info, "point", [0, 0, 0, 0])
x = core.MultiLatticeFermion(latt_info, num_mass * Ns * Nc)
for i in range(Ns * Nc):
    if num_mass == 1:
        quda.invertQuda(x[i].data_ptr, b[i].data_ptr, dirac.invert_param)
    else:
        quda.invertMultiShiftQuda(x[num_mass * i : num_mass * (i + 1)].data_ptrs, b[i].data_ptr, dirac.invert_param)

# for i in range(num_mass):
#     x_ = convert.multiFermionToPropagator(x[i::num_mass]) / (2 * rho)
#     x_ref = io.readKYUPropagator(
#         f"/home/zhangsh/workspace/tests/gwu_examples/08I_prop_full/gs8t8iwa165_052.grid.000000000_080808008.m{mass_ori[i]:.6f}", latt_info.global_size
#     )
#     print((x_ - x_ref).norm2() ** 0.5)
#     print(np.einsum("wtzyxijab,wtzyxijab->t", x_.data.conj(), x_.data))
#     print(np.einsum("wtzyxijab,wtzyxijab->t", x_ref.data.conj(), x_ref.data))

for i in range(num_mass):
    x_ = convert.multiFermionToPropagator(x[i::num_mass]) / (2 * rho)
    x_ref = io.readKYUPropagator(
        f"/home/zhangsh/workspace/tests/gwu_examples/08I_prop_high/multi_shift_noDef/gs8t8iwa165_052.grid.000000000_080808008.m{mass_ori[i]:.6f}", latt_info.global_size
    )
    print((x_ - x_ref).norm2() ** 0.5)
    print(np.einsum("wtzyxijab,wtzyxijab->t", x_.data.conj(), x_.data))
    print(np.einsum("wtzyxijab,wtzyxijab->t", x_ref.data.conj(), x_ref.data))