import numpy as np
import matplotlib.pyplot as plt
import kwant
import scipy.sparse
import scipy
from scipy import linalg as la
from scipy.sparse import linalg as sla
from types import SimpleNamespace
from conductivity import make_system

def kubo_conductivity(sys: kwant.Builder, params: dict, E_f: float = 0.0, energy: float=0.0, eta: float=1e-6)-> tuple:
    """
    Calculate the Kubo conductivity of the system at a given energy.

    energy: AC energy (frequency*hbar)

    equation source https://physics.stackexchange.com/questions/424158/what-is-the-difference-between-the-kubo-and-kubo-greenwood-formulas
    """
    
    final_sys = sys.finalized()
    
    # Get the Hamiltonian matrix
    h = final_sys.hamiltonian_submatrix(params=params, sparse=False)
    
    # Compute eigenvalues and eigenvectors
    eigs, states = np.linalg.eigh(h)
    states = states.T
    Vx = np.zeros_like(h, dtype=complex)
    Vy = np.zeros_like(h, dtype=complex)
    sites = list(final_sys.sites)
    for i, j in final_sys.graph:
        Vx[i, j] = 1j*(sites[i].pos[0] - sites[j].pos[0]) * h[i, j]
        Vy[i, j] = 1j*(sites[i].pos[1] - sites[j].pos[1]) * h[i, j]
    if not np.allclose(Vx- Vx.T.conj(), 0):
        print("making velocity operators hermitian")
        Vx += Vx.T.conj()
        Vy += Vy.T.conj()
    print("velocity operators: done")

    sigma_xx = 0.0 + 0.0j
    sigma_xy = 0.0 + 0.0j

    # transform the velocity operators to the eigenbasis
    Vx = states @ Vx @ states.conj().T
    Vy = states @ Vy @ states.conj().T

    for i in range(len(sites)):
        for j in range(len(sites)):
            if i == j: continue
            if (eigs[i]-E_f)*(eigs[j]-E_f) > 1e-12: continue
            Vx_ij = Vx[i, j]#np.vdot(states[i], Vx@states[j])
            Vy_ij = Vy[i, j]#np.vdot(states[i], Vy@states[j])
            sigma_xx = sigma_xx - 1/np.abs(eigs[j] - eigs[i]) * np.abs(Vx_ij)**2     / (energy + eigs[i] - eigs[j] + 1j*eta)
            sigma_xy = sigma_xy - 1/np.abs(eigs[j] - eigs[i]) * Vy_ij*np.conj(Vx_ij) / (energy + eigs[i] - eigs[j] + 1j*eta)
        print("i: ", i)
    sigma_xx *= 1j
    sigma_xy *= 1j

    return np.real(sigma_xx), np.real(sigma_xy)

if __name__ == "__main__":
    
    shapes = ["triangular-zz", "ribbon", "triangular-ac", "hexagon-zz", "hexagon-ac", "circle", "rectangle"]
    parameters = SimpleNamespace(phi=0.0, t_2=1e-12+0.2j, u_s=0.0, disorder=0.1)
    shape = shapes[0]
    diminsions = SimpleNamespace(r=5, l=8.2, N_edge=25, Lx = 5, L=10, W=7)
    diminsions.r = 4.5
    sys = make_system(1.0, shape, diminsions)
    kwant.plot(sys)

    final_sys = sys.finalized()

    parameters.t_2 = 1e-12 + 0.28j


    kc = kubo_conductivity(sys, params=dict(param=parameters), E_f=0.0, energy=0.0, eta=1e-9)

    print("sigma_xx:", kc[0])
    print("sigma_xy:", kc[1])

    # eigs, states = np.linalg.eigh(final_sys.hamiltonian_submatrix(params=dict(param=parameters), sparse=False))

