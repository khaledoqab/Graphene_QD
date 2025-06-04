import numpy as np
import matplotlib.pyplot as plt
import kwant
from types import SimpleNamespace

# Define the square lattice
lat = kwant.lattice.square(a=1.0, norbs=1)

def make_system(a, dimensions):
    def onsite(site, param):
        disorder = param.disorder * kwant.digest.gauss(repr(site), "")
        return disorder + param.m

    def hopping(site1, site2, param):
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        # return 2 * np.exp(2j*np.pi * param.phi * (x1 - x2) * (y1 + y2)/2)
        return np.exp(2j*np.pi * param.phi * (x1 - x2) *y2)

    sys = kwant.Builder()

    # Define rectangular region of L x W
    sys[lat.shape(lambda pos: 0 <= pos[0] < dimensions.L and 0 <= pos[1] < dimensions.W, (0, 0))] = onsite
    sys[lat.neighbors()] = hopping

    # Remove dangling sites
    for _ in range(2):
        for site in list(sys.sites()):
            if sum(1 for _ in sys.neighbors(site)) <= 1:
                del sys[site]

    return sys

def attach_leads(sys, dimensions):
    lead1 = kwant.Builder(kwant.TranslationalSymmetry((0, -1)))
    lead1[lat(0, 0)] = 0
    lead1[lat.neighbors()] = 2

    lead2 = kwant.Builder(kwant.TranslationalSymmetry((0, 1)))
    lead2[lat(dimensions.L - 1, dimensions.W - 1)] = 0
    lead2[lat.neighbors()] = 2

    # sys[lat(-1, 0)] = 0
    # sys[lat(-1, 0), lat(0, 0)] = -1
    sys.attach_lead(lead1)

    # sys[lat(dimensions.L - 1, dimensions.W)] = 0
    # sys[lat(dimensions.L - 1, dimensions.W), lat(dimensions.L - 1, dimensions.W - 1)] = -1
    sys.attach_lead(lead2)

    return sys

# Parameters and dimensions
dimensions = SimpleNamespace(L=10, W=10)
system_parameters = SimpleNamespace(phi=1/4, t_2=0.0, m=0.0, disorder=0.0)

# Build and finalize system
sys = make_system(lat, dimensions)
sys = attach_leads(sys, dimensions)
sys_fin = sys.finalized()
kwant.plot(sys_fin, site_size=0.1, show=True)
# Energy scan
energies = np.linspace(-4.0001, 0, 501)
transmissions = []

for energy in energies:
    smatrix = kwant.smatrix(sys_fin, energy, params=dict(param=system_parameters))
    transmissions.append(smatrix.transmission(0, 1))

plt.figure(figsize=(7, 5))
plt.plot(energies, transmissions, label="T(0→1)")
plt.xlabel("Energy")
plt.ylabel("Transmission coefficient")
plt.title("Transmission vs Energy")
plt.legend()
plt.grid(True)
plt.show()

# Flux scan
phis = np.linspace(0.1, 0.5, 500)
transmissions_phi = []

for phi in phis:
    params = dict(param=SimpleNamespace(
        phi=phi,
        t_2=0.0,
        m=0.0,
        disorder=0.0
    ))
    smatrix = kwant.smatrix(sys_fin, -2.0001, params=params)
    transmissions_phi.append(smatrix.transmission(0, 1))

plt.figure(figsize=(7, 5))
plt.plot(phis, transmissions_phi, label="T(0→1)")
plt.xlabel("Magnetic flux φ")
plt.ylabel("Transmission coefficient")
plt.title("Transmission vs Magnetic Flux")
plt.legend()
plt.grid(True)
plt.show()
