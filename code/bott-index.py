from haldane import make_system, lat, a_lat, b_lat, lat_neighbors_2
from utils2 import binary_search
import numpy as np
import matplotlib.pyplot as plt
import kwant
import scipy.linalg as la
from types import SimpleNamespace

def bott_index(sys: kwant.builder.FiniteSystem, system_parameters, ):
    h = sys.hamiltonian_submatrix(sparse=False, params=dict(param=system_parameters))
    N = len(h)
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    eigenstates = eigenvectors.T
    # i_0 = binary_search(eigenvalues, 0.0)

    P = np.zeros([len(eigenstates[0]),len(eigenstates[0])], dtype=complex)
    
    # N_occ = binary_search(eigenvalues, 0.0)
    N_occ = N//2

    for i in range(N_occ):
        # P += np.array([eigenstates[i]]).T@np.array([eigenstates[i].conj()])
        P += np.outer(eigenstates[i], eigenstates[i].conj())

    x_positions = np.array([s.pos[0] for s in sys.sites])
    x_norm = (x_positions-min(x_positions))/(max(x_positions)-min(x_positions))
    y_positions = np.array([s.pos[1] for s in sys.sites])
    y_norm = (y_positions-min(y_positions))/(max(y_positions)-min(y_positions))

    X = np.diag(x_norm)
    # X = np.diag(x_positions - min(x_positions))
    Y = np.diag(y_norm)
    # Y = np.diag(y_positions - min(y_positions))

    pXp = P@X@P
    pYp = P@Y@P

    U = la.expm(1j*2*np.pi*pXp)
    V = la.expm(1j*2*np.pi*pYp)
    UVUdVd = U@V@np.conj(U.T)@np.conj(V.T)
    bott = np.imag(np.trace(la.logm(UVUdVd)))/2/np.pi
    # bott = np.imag(np.sum(np.log(np.linalg.eigvals(UVUdVd))))/2/np.pi
    return bott

def main():
    shapes = ["triangular-zz", "ribbon", "triangular-ac", "hexagon-zz", "hexagon-ac", "circle", "rectangle"]
    parameters = SimpleNamespace(t_2=0.15*np.exp(-np.pi/3*1j), u_s=0.0, disorder=0.0)
    diminsions = SimpleNamespace(r=11, l=17, N_edge=14, Lx = 12, L=20, W=10)
    shape = shapes[6]

    # rect L=30, W=10: 8
    # rect L=40, W=10: 8
    # rect L=15, W=15: 9
    # rect L=20, W=20: 8
    # rect L=20, W=30: 8



    # tr -zz N_edge = 15: 0
    # tr -zz N_edge = 20: 0
    # tr -zz N_edge = 25: 0

    # ribbon W=10, L=5: 0
    # ribbon W=10, L=7: 0
    # ribbon W=10, L=10: 5
    # ribbon W=10, L=12: 6
    # ribbon W=10, L=15: 6
    # ribbon W=10, L=20: 7
    # ribbon W=10, L=25: 7
    # ribbon W=10, L=30: 8
    # ribbon W=10, L=35: 7
    # ribbon W=10, L=40: 7
    # ribbon W=10, L=45: 8

    # hex -ac r=6.0: 9
    # hex -ac r=7.8: 10
    # hex -ac r=9.3: 9
    # hex -ac r=11 : 11
    # hex -ac r=12.8: 10
    # hex -ac r=14.5: 10
    # hex -ac r=16.2: ??


    sys = make_system(1.0, shape, diminsions)
    fsys = sys.finalized()
    kwant.plot(sys, show=True)
    hamiltonian = sys.finalized().hamiltonian_submatrix(sparse=False, params=dict(param=parameters))
    eigs, _ = np.linalg.eigh(hamiltonian)
    plt.figure(figsize=(8, 6))
    plt.plot(eigs, marker='o')
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid()
    plt.show()
    masses = np.linspace(0, 1.5, 26)
    bott_indices = []
    for m in masses:
        parameters.u_s = m
        h = fsys.hamiltonian_submatrix(sparse=False, params=dict(param=parameters))
        bott = bott_index(fsys, system_parameters=parameters)
        print(f"m: {m}, bott index: {bott}")
        bott_indices.append(bott)

    plt.figure(figsize=(8, 6))
    plt.plot(masses, bott_indices, marker='o')
    plt.xlabel("Staggered potential (u_s)")
    plt.ylabel("Bott index")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

