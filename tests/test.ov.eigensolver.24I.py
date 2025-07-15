import numpy as np
from pyquda import enum_quda, pyquda as quda
from pyquda_utils import core, io

core.init([1, 1, 1, 4], [24, 24, 24, 64], 1, 1.0, backend="numpy", resource_path=".cache")
latt_info = core.getDefaultLattice()
kappa = 0.2
rho = 1.5
mass = 1 / (2 * kappa) - 4

dirac = core.getDirac(latt_info, mass, 1e-12, 100)
dirac.setPrecision(cuda=8, sloppy=8, precondition=8, eigensolver=8)
gauge = io.readKYUGauge("/public/ensemble/24I/rbc_conf_2464_m0.005_0.04_008545_hyp", latt_info.global_size)
dirac.loadGauge(gauge)
dirac.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_VERBOSE
dirac.invert_param.dslash_type = enum_quda.QudaDslashType.QUDA_OVERLAP_DSLASH
dirac.invert_param.overlap_invsqrt_tol = 1e-12

# H_wilson eigensolver parameters
hw_eig_param = quda.QudaEigParam()
hw_eig_param.n_ev = 300
hw_eig_param.n_kr = 350
hw_eig_param.n_conv = 300
hw_eig_param.tol = 1e-14
hw_eig_param.max_restarts = 1000
hw_eig_param.vec_infile = b"\0"
hw_eig_param.vec_outfile = b"\0"
# poly_acc settings
hw_eig_param.use_poly_acc = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
hw_eig_param.poly_deg = 100
hw_eig_param.a_min = 0.3
hw_eig_param.a_max = (1 + 8 * kappa)

quda.loadOverlapQuda(dirac.invert_param, hw_eig_param)

dirac.invert_param.mass = 0
# overlap eigensolver parameters(MdagMChiral)
ov_eig_param = quda.QudaEigParam()
ov_eig_param.invert_param = dirac.invert_param
ov_eig_param.n_ev = 200
ov_eig_param.n_kr = 400
ov_eig_param.n_conv = 200
ov_eig_param.tol = 1e-11
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
ov_eig_param.chirality = enum_quda.QudaChirality.QUDA_RIGHT_CHIRALITY
# poly_acc settings
ov_eig_param.use_poly_acc = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
ov_eig_param.poly_deg = 50
ov_eig_param.a_min = 0.1 / (2 * rho)
ov_eig_param.a_max = 1.0

evecs = core.MultiLatticeFermion(latt_info, ov_eig_param.n_ev)
evals = np.zeros(ov_eig_param.n_ev, "<c16")
quda.eigensolveQuda(evecs.data_ptrs, evals, ov_eig_param)