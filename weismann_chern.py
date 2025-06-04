import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg 
import kwant


def berry_curvature(syst, p, ks, num_filled_bands=1):
    """Berry curvature of a system.

    Parameters:
    -----------
    sys : kwant.Builder
        A 2D infinite system.
    p : dict
        The arguments expected by the system.
    ks : 1D array-like
        Values of momentum grid to be used for Berry curvature calculation.
    num_filled_bands : int
        The number of filled bands.

    Returns:
    --------
    bc : 2D array
        Berry curvature on each square in a `ks x ks` grid.
    """
    # Calculate an array of eigenvectors.
    B = np.array(syst.symmetry.periods).T
    A = B @ np.linalg.inv(B.T @ B)
    Kx, Ky = np.zeros([len(ks), len(ks)]), np.zeros([len(ks), len(ks)])
    syst = kwant.wraparound.wraparound(syst).finalized()

    def filled_states(kx, ky):
        k = np.array([kx, ky])
        kx, ky = np.linalg.solve(A, k)
        p.update(k_x=kx, k_y=ky)
        H = syst.hamiltonian_submatrix(params=p, sparse=False)
        return scipy.linalg.eigh(H)[1][:, :num_filled_bands]

    vectors = np.array(
        [[filled_states(kx, ky) for kx in ks] for ky in ks]
    )
    Kx = np.array(
        [[np.linalg.solve(A, np.array([kx, ky]))[0] for kx in ks] for ky in ks]
    )
    Ky = np.array(
        [[np.linalg.solve(A, np.array([kx, ky]))[1] for kx in ks] for ky in ks]
    )

    # The actual Berry curvature calculation
    vectors_x = np.roll(vectors, 1, 0)
    vectors_xy = np.roll(vectors_x, 1, 1)
    vectors_y = np.roll(vectors, 1, 1)

    shifted_vecs = [vectors, vectors_x, vectors_xy, vectors_y]

    v_shape = vectors.shape

    shifted_vecs = [i.reshape(-1, v_shape[-2], v_shape[-1]) for i in shifted_vecs]

    dets = np.ones(len(shifted_vecs[0]), dtype=complex)
    for vec, shifted in zip(shifted_vecs, np.roll(shifted_vecs, 1, 0)):
        dets *= [np.linalg.det(a.T.conj() @ b) for a, b in zip(vec, shifted)]
    bc = np.angle(dets).reshape(int(np.sqrt(len(dets))), -1)

    bc = (bc + np.pi / 2) % (np.pi) - np.pi / 2

    return bc, Kx, Ky

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
    syst[lat_neighbors_2] = lambda site1, site2, t_so: -1j*t_so

    # kwant.plot(syst, show=True)

    ks = np.linspace(-2*np.pi, 2*np.pi, 100)


    bc0, Kx, Ky = berry_curvature(syst, dict(t_so=0.001), ks, num_filled_bands=1)

    plt.imshow(
        berry_curvature(syst, dict(t_so=0.001), ks, num_filled_bands=1),
        cmap="RdBu"
    )
    plt.colorbar()
    plt.title("Berry curvature")
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.show()


    cherns = []
    socs = np.linspace(0, 1e-2, 20)
    for t_so_ in socs:
        bc = berry_curvature(syst, dict(t_so=t_so_), ks, num_filled_bands=1)
        cherns.append(bc.sum()/np.pi/2)
    plt.plot(socs, cherns)
    plt.xlabel("t_so")
    plt.ylabel("Chern number")
    # plt.title("Chern number as a function of t_so")
    plt.show()
