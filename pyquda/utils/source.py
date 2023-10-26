from typing import List, Literal, Union

from .. import mpi
from ..field import Nc, Ns, LatticeFermion, LatticePropagator, LatticeStaggeredFermion, LatticeStaggeredPropagator


def point(latt_size: List[int], t_srce: List[int], spin: int, color: int):
    Lx, Ly, Lz, Lt = latt_size
    x, y, z, t = t_srce
    gx, gy, gz, gt = mpi.coord
    b = LatticeFermion(latt_size) if spin is not None else LatticeStaggeredFermion(latt_size)
    if (
        gx * Lx <= x < (gx + 1) * Lx
        and gy * Ly <= y < (gy + 1) * Ly
        and gz * Lz <= z < (gz + 1) * Lz
        and gt * Lt <= t < (gt + 1) * Lt
    ):
        eo = ((x - gx * Lx) + (y - gy * Ly) + (z - gz * Lz) + (t - gt * Lt)) % 2
        if spin is not None:
            b.data[eo, t - gt * Lt, z - gz * Lz, y - gy * Ly, (x - gx * Lx) // 2, spin, color] = 1
        else:
            b.data[eo, t - gt * Lt, z - gz * Lz, y - gy * Ly, (x - gx * Lx) // 2, color] = 1

    return b


def wall(latt_size: List[int], t_srce: int, spin: int, color: int):
    Lx, Ly, Lz, Lt = latt_size
    gx, gy, gz, gt = mpi.coord
    t = t_srce
    b = LatticeFermion(latt_size) if spin is not None else LatticeStaggeredFermion(latt_size)
    if gt * Lt <= t < (gt + 1) * Lt:
        if spin is not None:
            b.data[:, t - gt * Lt, :, :, :, spin, color] = 1
        else:
            b.data[:, t - gt * Lt, :, :, :, color] = 1

    return b


def momentum(latt_size: List[int], t_srce: int, spin: int, color: int, phase):
    Lx, Ly, Lz, Lt = latt_size
    gx, gy, gz, gt = mpi.coord
    t = t_srce
    b = LatticeFermion(latt_size) if spin is not None else LatticeStaggeredFermion(latt_size)
    if gt * Lt <= t < (gt + 1) * Lt:
        if spin is not None:
            b.data[:, t - gt * Lt, :, :, :, spin, color] = phase[:, t - gt * Lt, :, :, :]
        else:
            b.data[:, t - gt * Lt, :, :, :, color] = phase[:, t - gt * Lt, :, :, :]

    return b


def gaussian3(latt_size: List[int], t_srce: List[int], spin: int, color: int, rho: float, nsteps: int):
    from .. import core
    from ..enum_quda import QudaDslashType

    _b = point(latt_size, t_srce, None, color)
    dslash = core.getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
    dslash.invert_param.dslash_type = QudaDslashType.QUDA_LAPLACE_DSLASH
    alpha = 1 / (4 * nsteps / rho**2 - 6)
    core.quda.performWuppertalnStep(_b.data_ptr, _b.data_ptr, dslash.invert_param, nsteps, alpha)

    if spin is not None:
        b = LatticeFermion(latt_size)
        b.data[:, :, :, :, :, spin, :] = _b.data
    else:
        b = LatticeStaggeredFermion(latt_size, _b.data)

    return b


def gaussian2(latt_size: List[int], t_srce: List[int], spin: int, color: int, rho: float, nsteps: int, xi: float):
    from .. import core
    from ..enum_quda import QudaDslashType

    def _Laplacian(src, aux, sigma, invert_param):
        # aux = -kappa * Laplace * src + src
        core.quda.MatQuda(aux.data_ptr, src.data_ptr, invert_param)
        src.data = -6 * sigma * src.data + aux.data

    _b = point(latt_size, t_srce, None, color)
    _c = LatticeStaggeredFermion(latt_size)

    # use mass to get specific kappa = -xi * rho**2 / 4 / nsteps
    kappa = -(rho**2) / 4 / nsteps * xi
    mass = 1 / (2 * kappa) - 4
    dslash = core.getDslash(latt_size, mass, 0, 0, anti_periodic_t=False)
    dslash.invert_param.dslash_type = QudaDslashType.QUDA_LAPLACE_DSLASH
    for _ in range(nsteps):
        # (rho**2 / 4) here aims to achieve the same result with Chroma
        _Laplacian(_b, _c, rho**2 / 4 / nsteps, dslash.invert_param)
    _c = None

    if spin is not None:
        b = LatticeFermion(latt_size)
        b.data[:, :, :, :, :, spin, :] = _b.data
    else:
        b = LatticeStaggeredFermion(latt_size, _b.data)

    return b


def gaussian(latt_size: List[int], t_srce: List[int], spin: int, color: int, rho: float, nsteps: int, xi: float):
    from .. import core
    from ..enum_quda import QudaDslashType, QudaParity

    def _Laplacian(src, aux, sigma, xi, invert_param):
        aux.data[:] = 0
        core.quda.dslashQuda(aux.even_ptr, src.odd_ptr, invert_param, QudaParity.QUDA_EVEN_PARITY)
        core.quda.dslashQuda(aux.odd_ptr, src.even_ptr, invert_param, QudaParity.QUDA_ODD_PARITY)
        aux.even -= src.odd
        aux.odd -= src.even
        aux.data *= xi
        src.data = (1 - sigma * 6) * src.data + sigma * aux.data

    Lx, Ly, Lz, Lt = latt_size
    gx, gy, gz, gt = mpi.coord
    x, y, z, t = t_srce
    _b = LatticeStaggeredFermion(latt_size)
    _c = LatticeStaggeredFermion(latt_size)

    if (
        gx * Lx <= x < (gx + 1) * Lx
        and gy * Ly <= y < (gy + 1) * Ly
        and gz * Lz <= z < (gz + 1) * Lz
        and gt * Lt <= t < (gt + 1) * Lt
    ):
        eo = ((x - gx * Lx) + (y - gy * Ly) + (z - gz * Lz) + (t - gt * Lt)) % 2
        _b.data[eo, t - gt * Lt, z - gz * Lz, y - gy * Ly, (x - gx * Lx) // 2, color] = 1

    dslash = core.getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
    dslash.invert_param.dslash_type = QudaDslashType.QUDA_LAPLACE_DSLASH
    for _ in range(nsteps):
        # (rho**2 / 4) aims to achieve the same result with Chroma
        _Laplacian(_b, _c, rho**2 / 4 / nsteps, xi, dslash.invert_param)
    _c = None

    if spin is not None:
        b = LatticeFermion(latt_size)
        b.data[:, :, :, :, :, spin, :] = _b.data
    else:
        b = LatticeStaggeredFermion(latt_size, _b.data)

    return b


def colorvector(latt_size: List[int], t_srce: int, phase):
    Lx, Ly, Lz, Lt = latt_size
    gx, gy, gz, gt = mpi.coord
    t = t_srce
    b = LatticeStaggeredFermion(latt_size)
    if gt * Lt <= t < (gt + 1) * Lt:
        b.data[:, t - gt * Lt, :, :, :, :] = phase[:, t - gt * Lt, :, :, :, :]
    return b


def source(
    latt_size: List[int],
    source_type: str,
    t_srce: Union[int, List[int]],
    spin: int,
    color: int,
    source_phase=None,
    rho: float = 0.0,
    nsteps: int = 0,
    xi: float = 1.0,
):
    if source_type.lower() == "point":
        return point(latt_size, t_srce, spin, color)
    elif source_type.lower() == "wall":
        return wall(latt_size, t_srce, spin, color)
    elif source_type.lower() == "momentum":
        return momentum(latt_size, t_srce, spin, color, source_phase)
    elif source_type.lower() == "gaussian":
        return gaussian(latt_size, t_srce, None, color, rho, nsteps, xi)
    elif source_type.lower() == "colorvector":
        return colorvector(latt_size, t_srce, source_phase)
    else:
        raise NotImplementedError(f"{source_type} source is not implemented yet.")


def source12(
    latt_size: List[int],
    source_type: Literal["point", "wall", "momentum", "gaussian", "colorvector"],
    t_srce: Union[int, List[int]],
    source_phase=None,
    rho: float = 0.0,
    nsteps: int = 0,
    xi: float = 1.0,
):
    Lx, Ly, Lz, Lt = latt_size
    Vol = Lx * Ly * Lz * Lt

    b12 = LatticePropagator(latt_size)
    data = b12.data.reshape(Vol, Ns, Ns, Nc, Nc)
    if source_type.lower() in ["colorvector"]:
        b = source(latt_size, source_type, t_srce, None, None, source_phase)
        for color in range(Nc):
            for spin in range(Ns):
                data[:, spin, spin, :, color] = b.data.reshape(Vol, Nc)
    elif source_type.lower() in ["gaussian"]:
        for color in range(Nc):
            b = source(latt_size, source_type, t_srce, None, color, source_phase, rho, nsteps, xi)
            for spin in range(Ns):
                data[:, spin, spin, :, color] = b.data.reshape(Vol, Nc)
    else:
        for color in range(Nc):
            for spin in range(Ns):
                b = source(latt_size, source_type, t_srce, spin, color, source_phase)
                data[:, :, spin, :, color] = b.data.reshape(Vol, Ns, Nc)

    return b12


def source3(
    latt_size: List[int],
    source_type: Literal["point", "wall", "momentum", "gaussian", "colorvector"],
    t_srce: Union[int, List[int]],
    source_phase=None,
    rho: float = 0.0,
    nsteps: int = 0,
    xi: float = 1.0,
):
    Lx, Ly, Lz, Lt = latt_size
    Vol = Lx * Ly * Lz * Lt

    b3 = LatticeStaggeredPropagator(latt_size)
    data = b3.data.reshape(Vol, Nc, Nc)

    for color in range(Nc):
        b = source(latt_size, source_type, t_srce, None, color, source_phase, rho, nsteps, xi)
        data[:, :, color] = b.data.reshape(Vol, Nc)

    return b3
