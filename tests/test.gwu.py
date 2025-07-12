from pyquda_comm.field import LatticeInfo
from pyquda_utils import io, convert, source
from pyquda_plugins import pygwu
import numpy as np

# # [24, 24, 24, 64]
# pygwu.init([24, 24, 24, 64])
# latt_info = LatticeInfo([24, 24, 24, 64], 1, 1.0)
# gauge = io.readKYUGauge("/public/ensemble/24I/rbc_conf_2464_m0.005_0.04_008545_hyp", latt_info.global_size)

# overlap = pygwu.Overlap(latt_info)
# overlap.buildHWilson(gauge, kappa=0.2)
# overlap.buildHWilsonEigen(300, 1e-13, 50, 500, 100, 0.3, 1394987439)
# overlap.buildOverlap(ov_poly_prec=1e-12, ov_use_fp32=1)
# overlap.loadOverlapEigen(200, 1e-10, "/public/ensemble/24I/rbc_conf_2464_m0.005_0.04_008545_hyp.half.overlap.eigensystem", True)

# b = source.multiFermion(latt_info, "point", [0, 0, 0, 0])
# b = pygwu.multiFermionFromDiracPauli(b)
# x = overlap.invert(b, masses=[0.0160, 0.0260, 0.0460], tol=1e-8, maxiter=500, one_minus_half_d=1, mode=0b11)

# [8, 8, 8, 8]
pygwu.init([8, 8, 8, 8])
latt_info = LatticeInfo([8, 8, 8, 8], 1, 1.0)
gauge = io.readKYUGauge("/home/zhangsh/workspace/gwu-qcd-ck/examples/gs8t8iwa165_052", latt_info.global_size)

overlap = pygwu.Overlap(latt_info)
overlap.buildHWilson(gauge, kappa=0.2)
# overlap.loadHWilsonEigen(100, 1e-13, "/home/zhangsh/workspace/tests/gwu_examples/gs8t8iwa165_052.Hwilson.eigensystem", False)
overlap.buildHWilsonEigen(100, 1e-13, 300, 500, 50, 0.18, 1394987439)
overlap.buildOverlap(ov_poly_prec=1e-12, ov_use_fp32=1)
overlap.loadOverlapEigen(50, 1e-10, "/home/zhangsh/workspace/tests/gwu_examples/gs8t8iwa165_052.half.overlap.eigensystem", True)

b = source.multiFermion(latt_info, "point", [0, 0, 0, 0])
b = pygwu.multiFermionFromDiracPauli(b)
x = overlap.invert(b, masses=[0.1, 0.2, 0.5], tol=1e-8, maxiter=500, one_minus_half_d=1, mode=0b11)

# x_ = convert.multiFermionToPropagator(x[0:12])

# x_ref = io.readKYUPropagator(
#     "/home/zhangsh/workspace/tests/gwu_examples/08I_prop_full/gs8t8iwa165_052.grid.000000000_080808008.m0.100000", latt_info.global_size
# )

# print("norm2 between x_ and x_ref: ", (x_ - x_ref).norm2() ** 0.5)

# x_arr = x_.lexico()
# x_ref_arr = x_ref.lexico()
# print("x_arr.shape : ", x_arr.shape)
# print("x_ref_arr.shape : ", x_ref_arr.shape)
# print("x_arr[0, 0, 0, 0, 0, :, 0, :, 0] : \n", x_arr[0, 0, 0, 0, :, 0, :, 0])
# print("x_ref_arr[0, 0, 0, 0, 0, :, 0, :, 0] : \n", x_ref_arr[0, 0, 0, 0, :, 0, :, 0])