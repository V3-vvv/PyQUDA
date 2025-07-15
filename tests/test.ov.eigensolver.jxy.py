from os import path
import numpy as np
import cupy as cp
from pyquda import enum_quda, pyquda as quda
from pyquda_comm.pointer import ndarrayPointer
from pyquda_utils import core, io, gamma, source
from pyquda.field import LatticeFermion, MultiLatticeFermion, LatticePropagator, LatticeInfo, evenodd, Nc, Ns
from pyquda_plugins import pygwu
from pyquda_utils import convert

core.init(None, [24, 24, 24, 64], 1, 1.0, backend="numpy", resource_path=".cache")
latt_info = core.getDefaultLattice()
kappa = 0.2
rho = 1.5
num_eigen = 50

mass_ori = np.array([0.1, 0.2, 0.5], "<f8")
mass = mass_ori / (2 * rho)
offset = mass**2 / (1 - mass**2)
num_mass = 3

dirac = core.getDirac(latt_info, mass[0], 1e-12, 100)
dirac.setPrecision(cuda=8, sloppy=8, precondition=8, eigensolver=8)
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

eig_param_hw = quda.QudaEigParam()
eig_param_hw.use_poly_acc = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
eig_param_hw.poly_deg = 100
eig_param_hw.a_min = 0.3
eig_param_hw.a_max = 1 + 8 * kappa
eig_param_hw.n_ev = 300
eig_param_hw.n_kr = 350
eig_param_hw.n_conv = 300
eig_param_hw.tol = 1e-14
eig_param_hw.max_restarts = 1000
eig_param_hw.vec_infile = b"\0"
eig_param_hw.vec_outfile = b"\0"

gauge = io.readKYUGauge("/public/ensemble/24I/rbc_conf_2464_m0.005_0.04_008545_hyp", latt_info.global_size)
dirac.loadGauge(gauge)
quda.loadOverlapQuda(dirac.invert_param, eig_param_hw)

n_ev = 200
n_kr = 400
dirac.invert_param.mass = 0
eig_param = quda.QudaEigParam()
eig_param.invert_param = dirac.invert_param
eig_param.eig_type = enum_quda.QudaEigType.QUDA_EIG_BLK_TR_LANCZOS
eig_param.use_dagger = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
eig_param.use_norm_op = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
eig_param.use_pc = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
eig_param.compute_gamma5 = enum_quda.QudaBoolean.QUDA_BOOLEAN_FALSE
eig_param.spectrum = enum_quda.QudaEigSpectrumType.QUDA_SPECTRUM_SR_EIG
eig_param.use_poly_acc = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
eig_param.poly_deg = 50
eig_param.a_min = 0.1 / (2 * rho)
eig_param.a_max = 1.0
eig_param.n_ev = n_ev
eig_param.n_kr = n_kr
eig_param.n_conv = n_ev
eig_param.tol = 1e-11
eig_param.max_restarts = 100
eig_param.vec_infile = b""
eig_param.vec_outfile = b""
eig_param.chirality = enum_quda.QudaChirality.QUDA_RIGHT_CHIRALITY

eig_param.batched_rotate = 1
eig_param.compute_evals_batch_size = 1

evecs = core.MultiLatticeFermion(latt_info, n_ev)
evals = np.zeros((n_ev), "<c16")
quda.eigensolveQuda(evecs.data_ptrs, evals, eig_param)