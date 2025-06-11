import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse.linalg as sla
import warnings
warnings.filterwarnings("ignore")
import kwant
import tinyarray


# define the lattice
a = 1.0
lat = kwant.lattice.honeycomb(a=a, norbs=2)
a_lat, b_lat = lat.sublattices

# define the hoppings
nnn_hoppings_a = (((-1, 0), a_lat, a_lat), ((0, 1), a_lat, a_lat), ((1, -1), a_lat, a_lat))
nnn_hoppings_b = (((1, 0), b_lat, b_lat), ((0, -1), b_lat, b_lat), ((-1, 1), b_lat, b_lat))
nnn_hoppings_all = nnn_hoppings_a + nnn_hoppings_b
lat_neighbors_2 = [kwant.builder.HoppingKind(*hop) for hop in nnn_hoppings_all]

# pauli matrices
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])
sigma_0 = tinyarray.array([[1, 0], [0, 1]])

def binary_search(arr, x):
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            l = mid + 1
        else:
            r = mid - 1
    return l

# define the onsite potential
def onsite(site, param):
    eps = param.m if site.family == lat.sublattices[0] else -param.m
    eps_imp = param.U_imp if np.random.rand() < param.p_imp else 0
    return (eps + eps_imp)*sigma_0

# define the nearest-neighbor hopping term
def nn_hopping(site1, site2, param):
    x1, y1 = site1.pos
    x2, y2 = site2.pos
    rashba_term = 0
    if param.t_rsh is not None:
        d_hat = np.array([x2 - x1, y2 - y1])
        d_hat = d_hat/np.linalg.norm(d_hat)
        rashba_term = param.t_rsh*d_hat[0]*sigma_x - param.t_rsh*d_hat[1]*sigma_y
    return param.t*np.exp(-1j*param.B/2*(x2 - x1)*(y1 + y2))*sigma_0 + 1j*rashba_term

# define the next-nearest neighbor hopping term
def nnn_hopping(site1, site2, param):
    x1, y1 = site1.pos
    x2, y2 = site2.pos
    return 1j*param.t_prime*np.exp(-1j*param.B/2*(x2 - x1)*(y1 + y2))*sigma_z + param.t2*sigma_0



# define the system
def make_triangular_zigzag_system(N_edge):
    sys = kwant.Builder()
    # add the lattice to the system
    # for i in range(N_edge):
    #     for j in range(N_edge- i):
    #         sys[a_lat(i, j)] = onsite
    #         sys[b_lat(i, j)] = onsite
    
    Lx, Ly = 20, 10
    sys[lat.shape((lambda pos: 0<=pos[0] < Lx and 0<=pos[1] < Ly), (0, 0))] = onsite

    sys[lat.neighbors(n=1)] = nn_hopping

    sites = list(sys.sites())
    # Count the number of connections (hopping terms) for the site
    # make sure that no lattice point is connected to more than one site

    for __ in range(2):
        sites = list(sys.sites())
        for s in sites:
            num_connections = sum(1 for _ in sys.neighbors(s))
            if num_connections == 1 or num_connections == 0: del sys[s]

    sys[lat_neighbors_2] = nnn_hopping

    return sys

# define the parameters
class Parameters:
    def __init__(self, t, m, U_imp, p_imp, B, t_prime, t2, t_rsh=None):
        self.t = t
        self.m = m
        self.U_imp = U_imp
        self.p_imp = p_imp
        self.B = B
        self.t_prime = t_prime
        self.t2 = t2
        self.t_rsh = t_rsh

def spin_bott_index(sys, params):
    # finalize the system
    sys = sys.finalized()
    # get the Hamiltonian matrix
    H = sys.hamiltonian_submatrix(params=dict(param = params))
    n = len(H)
    # get the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(H)
    eigenstates = eigvecs.T

    P = np.zeros([n, n], dtype=complex)

    N_occ = binary_search(eigvals, 0)
    eff_pauli_z = np.kron(np.eye(n//2), sigma_z)

    for i in range(N_occ):
        P = P + np.array([eigenstates[i]]).T@np.array([eigenstates[i]]).conj()
    
    Pz = P@eff_pauli_z@P

    # plt.plot(np.sort(np.real(np.linalg.eigvals(Pz))), 'o')
    # plt.xlabel("index")
    # plt.ylabel("Eigenvalues")
    # plt.title("Eigenvalues of Pz")
    # plt.show()

    spm, phi_pm = np.linalg.eig(Pz)
    phis = phi_pm.T

    Pp = np.zeros_like(Pz, dtype=complex)
    Pm = np.zeros_like(Pz, dtype=complex)
    n_m, n_p = 0, 0
    for i in range(len(spm)):
        if np.allclose(spm[i], 0): pass
        elif spm[i] > 0: 
            Pp += np.array([phis[i]]).T@np.array([phis[i]]).conj()
            n_p += 1
        else: 
            Pm += np.array([phis[i]]).T@np.array([phis[i]]).conj()
            n_m += 1
    print(f"N_occ: {N_occ} n_p: {n_p}, n_m: {n_m}")

    x_positions = np.array([s.pos[0] for s in sys.sites])
    x_norm = (x_positions-min(x_positions))/(max(x_positions)-min(x_positions))
    y_positions = np.array([s.pos[1] for s in sys.sites])
    y_norm = (y_positions-min(y_positions))/(max(y_positions)-min(y_positions))
    
    X = np.kron(np.diag(x_norm), np.eye(2))
    Y = np.kron(np.diag(y_norm), np.eye(2))

    Up = la.expm(2*np.pi*1j*Pp@X@Pp) #+ 1 - Pp
    Um = la.expm(2*np.pi*1j*Pm@X@Pm) #+ 1 - Pm
    Vp = la.expm(2*np.pi*1j*Pp@Y@Pp) #+ 1 - Pp
    Vm = la.expm(2*np.pi*1j*Pm@Y@Pm) #+ 1 - Pm

    Bp = np.imag(np.sum(np.log(np.linalg.eigvals(Up@Vp@Up.T.conj()@Vp.T.conj()))))
    Bm = np.imag(np.sum(np.log(np.linalg.eigvals(Um@Vm@Um.T.conj()@Vm.T.conj()))))
    bott_s = (Bp - Bm)/(4*np.pi)
    return bott_s

if __name__ == "__main__":
    # Example usage
    parameters = Parameters(
        t=1.0,
        m=2.0,
        U_imp=0.0,
        p_imp=0.0,
        B=0.0,
        t_prime=0.3,
        t2=0.0,
        t_rsh=0.25
    )
    
    triangle = make_triangular_zigzag_system(15)
    # print(spin_bott_index(triangle, parameters))
    
    kwant.plot(triangle, show=False)
    # plt.savefig("zigzag.png")
    plt.show()

    bott_index = []
    m_s = np.linspace(0.0, 2.5, 27)
    for m_ in m_s:
        parameters = Parameters(t=1.0, m=m_, U_imp=0.0, p_imp=0.0, B=0.0, t_prime=0.3, t2=0.0, t_rsh=0.25)
        sbi = spin_bott_index(triangle, parameters)
        print(f"m: {m_}, spin bott index: {sbi}")
        bott_index.append(sbi)
    plt.plot(m_s, bott_index)
    plt.xlabel("t'")
    plt.ylabel("Spin Bott Index")
    plt.title("Spin Bott Index vs t'")
    plt.grid()
    # plt.savefig(f"spin_bott_indexB-0--m-0--t2-0-Uimp-0--p-0--Nedge-15--trzz-1.png")
    plt.show()