from time import perf_counter

import numpy as np

from pyquda import init
from pyquda.utils import io
from pyquda.field import LatticeInfo, LatticeGauge, Nc

from hmc import HMC

init(resource_path=".cache")
latt_info = LatticeInfo([16, 16, 16, 32])

gauge = LatticeGauge(latt_info)


u_0 = 0.855453
beta = 6.20 * u_0**4
input_path = [
    [0, 1, 4, 5],
    [0, 2, 4, 6],
    [0, 3, 4, 7],
    [1, 2, 5, 6],
    [1, 3, 5, 7],
    [2, 3, 6, 7],
]
input_coeff_ = [
    -1 / u_0**4,
    -1 / u_0**4,
    -1 / u_0**4,
    -1 / u_0**4,
    -1 / u_0**4,
    -1 / u_0**4,
]
input_coeff = [val * beta / Nc for val in input_coeff_]

input_path2 = [
    [
        [0, 1, 4, 5],
        [0, 5, 4, 1],
        [0, 2, 4, 6],
        [0, 6, 4, 2],
        [0, 3, 4, 7],
        [0, 7, 4, 3],
    ],
    [
        [1, 4, 5, 0],
        [1, 0, 5, 4],
        [1, 2, 5, 6],
        [1, 6, 5, 2],
        [1, 3, 5, 7],
        [1, 7, 5, 3],
    ],
    [
        [2, 4, 6, 0],
        [2, 0, 6, 4],
        [2, 5, 6, 1],
        [2, 1, 6, 5],
        [2, 3, 6, 7],
        [2, 7, 6, 3],
    ],
    [
        [3, 4, 7, 0],
        [3, 0, 7, 4],
        [3, 5, 7, 1],
        [3, 1, 7, 5],
        [3, 6, 7, 2],
        [3, 2, 7, 6],
    ],
]
input_coeff2_ = [
    1 / u_0**4,
    1 / u_0**4,
    1 / u_0**4,
    1 / u_0**4,
    1 / u_0**4,
    1 / u_0**4,
]
input_coeff2 = [val * beta / Nc for val in input_coeff2_]


hmc = HMC(latt_info)
hmc.setVerbosity(0)
hmc.initialize()

start = 0
stop = 2000
warm = 500
save = 5

print("\n" f"Trajectory {start}:\n" f"plaquette = {hmc.plaquette()}\n")

t = 1.0
n_steps = 10
for i in range(start, stop):
    s = perf_counter()

    hmc.gaussMom(i)

    kinetic_old = hmc.actionMom()
    potential_old = hmc.actionGauge(input_path, input_coeff)
    energy_old = kinetic_old + potential_old

    dt = t / n_steps
    for _ in range(n_steps):
        hmc.updateMom(input_path2, input_coeff2, dt / 2)
        hmc.updateGauge(dt)
        hmc.updateMom(input_path2, input_coeff2, dt / 2)

    hmc.reunitGauge(1e-15)

    kinetic = hmc.actionMom()
    potential = hmc.actionGauge(input_path, input_coeff)
    energy = kinetic + potential

    accept = np.random.rand() < np.exp(energy_old - energy)
    if accept or i < warm:
        hmc.saveGauge(gauge)
    else:
        hmc.loadGauge(gauge)

    print(
        f"Trajectory {i + 1}:\n"
        f"plaquette = {hmc.plaquette()}\n"
        f"P_old = {potential_old}, K_old = {kinetic_old}\n"
        f"P = {potential}, K = {kinetic}\n"
        f"Delta_P = {potential - potential_old}, Delta_K = {kinetic - kinetic_old}\n"
        f"Delta_E = {energy - energy_old}\n"
        f"acceptance rate = {min(1, np.exp(energy_old - energy))*100:.2f}%\n"
        f"accept? {accept or i < warm}\n"
        f"HMC time = {perf_counter() - s:.3f} secs\n"
    )

    if (i + 1) % save == 0:
        io.writeKYUGauge(f"./DATA/cfg/cfg_{i + 1}.kyu", gauge)
