import numpy as np
import matplotlib.pyplot as plt
import kwant

DK = 1e-2
K_GRID = np.meshgrid(np.arange(-2*np.pi, 2*np.pi, DK), np.arange(-2*np.pi, 2*np.pi, DK))

def chern_number(syst, parameters):

    B = np.array(syst.symmetry.periods)
    kx, ky = K_GRID
    n, m= kx.shape

    k_sys = kwant.wraparound.wraparound(syst).finalized()
    parameters.update(k_x=0, k_y=0)
    print(B.T)
    print(kx[0, 0], ky[0, 0])
    dat = np.zeros((n, m, 2, 2), dtype=complex)
    for i in range(n):
        for j in range(m):
            kx1, ky1 = B.T @ np.array([kx[i, j], ky[i, j]])
            parameters.update(k_x=kx1, k_y=ky1)
            h = k_sys.hamiltonian_submatrix(params=parameters, sparse=False)
            vecs = np.linalg.eigh(h)[1][:, :1]
            dat[i, j, :, :] = vecs
    print(dat[0, 0])
    Phis = np.zeros((n), dtype=complex)
    for i in range(n):
        prod = 1.0
        for p in range(m-1):
            det = np.linalg.det(dat[p, 0].T @ (dat[p+1, 0]))
            prod *= det
        Phis[i] = (np.angle(prod) + np.pi/2) % (np.pi) - np.pi/2

    return np.sum(Phis)/np.pi/2


if __name__ == "__main__":
    lat = kwant.lattice.honeycomb(a=1.0, norbs=1)
    a, b = lat.sublattices

    nnn_hoppings_a = (((-1, 0), a, a), ((0,  1), a, a), ((1, -1), a, a))
    nnn_hoppings_b = ((( 1, 0), b, b), ((0, -1), b, b), ((-1, 1), b, b))
    nnn_hoppings_all = nnn_hoppings_a + nnn_hoppings_b
    lat_neighbors_2 = [kwant.builder.HoppingKind(*hop) for hop in nnn_hoppings_all]

    syst = kwant.Builder(kwant.TranslationalSymmetry(*lat.prim_vecs))
    syst[a(0, 0)] = 0
    syst[b(0, 0)] = 0
    syst[lat.neighbors()] = -1
    syst[lat_neighbors_2] = -1j*0.1

    kwant.plot(syst, show=True)
    print(chern_number(syst, {}))