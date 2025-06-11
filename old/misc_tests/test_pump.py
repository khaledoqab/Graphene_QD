import kwant
import kwant.physics
import numpy as np
import matplotlib.pyplot as plt
L = 60

sys = kwant.Builder(kwant.TranslationalSymmetry([-L]))
lat = kwant.lattice.chain(norbs=1)
mu = 0.0
t = 1.0
A = 3.0

omega = 5*2*np.pi/L
# phi = 0.0

def onsite(site, phi):
	return 2*t - mu + A*(1 - np.cos(omega*site.pos[0] + phi))

sys[lat.shape((lambda x: True), [0])] = onsite
sys[lat.neighbors()] = -t

wire = kwant.Builder()
wire.fill(sys, shape = (lambda site: 0<= site.pos[0]<L), start=[0])
lead = kwant.Builder(kwant.TranslationalSymmetry([-1]))
lead[lat(0)] = lambda site: 2 * t - 0.0
lead[lat.neighbors()] = -t

wire.attach_lead(lead, add_cells=80)
wire.attach_lead(lead.reversed(), add_cells=80)
wire = wire.finalized()


# kwant.plot(wire)
# plt.show()
bands = []
# for i in range(1000):
# 	bands.append(kwant.physics.Bands(wire.finalized(), params=dict(phi=i/500*np.pi))(0))
# plt.plot(bands)
# plt.show()
for phi in np.linspace(0, 2*np.pi, 256):
	en = np.linalg.eigh(wire.hamiltonian_submatrix(params=dict(phi=phi)))[0]
	bands.append(en)
print(np.array(bands).shape)
plt.plot(bands)
plt.show()

psi = kwant.wave_function(wire, energy=0.01, params=dict(phi=0.0))(1)
es, psis = np.linalg.eigh(wire.hamiltonian_submatrix(params=dict(phi=0.0)))
psis = psis.T[np.argsort(es)]
es = np.sort(es)
plt.scatter(range(len(es)), es)
plt.show()

print(psi.shape)
plt.plot(np.abs(psis[35])**2)
plt.plot(range(80, 140), (1 - np.cos(omega*np.arange(0, L)))/10)
plt.show()
