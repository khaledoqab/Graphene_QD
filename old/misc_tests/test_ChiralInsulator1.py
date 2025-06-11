import numpy as np
import matplotlib.pyplot as plt
import kwant
import tinyarray
from types import SimpleNamespace
import sys

lat = kwant.lattice.square(a=1, norbs=1)
s0, sx, sy, sz = tinyarray.array([[1, 0], [0, 1]]), tinyarray.array([[0, 1], [1, 0]]), tinyarray.array([[0, -1j], [1j, 0]]), tinyarray.array([[1, 0], [0, -1]])
sys = kwant.Builder(kwant.TranslationalSymmetry(*lat.prim_vecs))

def onsite(site, mu):
    return -mu * sz


def hopx(site1, site2, delta, t):
    return -0.5j * delta * sy - t * sz


def hopy(site1, site2, gamma):
    x_term = -gamma * sx
    z_term = -gamma * sz
    y_term = - gamma * sy
    ax, ay, az = 0., 0.15j, 1.0
    return az*z_term  + ax*x_term + ay*y_term

sys[lat(0, 0)] = onsite(0, 0)
sys[kwant.HoppingKind((1, 0), lat)] = hopx
sys[kwant.HoppingKind((0, 1), lat)] = hopy


# Define the parameters
p = dict(t=1.0, delta=.3, gamma=-0.5, mu=None)
mus = np.linspace(-2, 0, 11)

ribbon = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))
ribbon.fill(sys, lambda site: 0<= site.pos[1] <= 15, start=(0, 0))

kwant.plotter.bands(ribbon.finalized(), params=p, momenta=256)
plt.show()