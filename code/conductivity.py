import numpy as np
import matplotlib.pyplot as plt
import kwant
import scipy.sparse
import tinyarray
import scipy
from scipy import linalg as la
from scipy.sparse import linalg as sla
from matplotlib.patches import ConnectionPatch
from types import SimpleNamespace
from copy import deepcopy
from utils2 import binary_search, calculate_participation_ratio
import argparse


# lat = kwant.lattice.honeycomb(a=1.0, norbs=2)
# a_lat, b_lat = lat.sublattices
lat = kwant.lattice.general([[1, 0], [1/2, np.sqrt(3)/2]],  # lattice vectors
                      [[0, 0], [0, 1/np.sqrt(3)]], norbs=1)  # Coordinates of the sites
a_lat, b_lat = lat.sublattices

nnn_hoppings_a = (((-1, 0), a_lat, a_lat), ((0, 1), a_lat, a_lat), ((1, -1), a_lat, a_lat))
nnn_hoppings_b = (((1, 0), b_lat, b_lat), ((0, -1), b_lat, b_lat), ((-1, 1), b_lat, b_lat))
nnn_hoppings_all = nnn_hoppings_a + nnn_hoppings_b
lat_neighbors_2 = [kwant.builder.HoppingKind(*hop) for hop in nnn_hoppings_all]


def atan(y, x):
    ans = np.arctan2(y, x)
    if type ( x + 0.0) == type(y+0.0) and type(x+0.0) in [float, np.float64, np.float128, np.float_]:

        if ans < 0: return 2*np.pi + ans
        else: return ans

    elif type(x) == type(y) and type(x) == type(np.array([])) :
        ans[ans < 0] += 2*np.pi
        return ans

    else: raise TypeError("atan is getting neither floats nor arrays")

def Regular_Polygon(r, n = 3, start = 0):
    if n < 3: raise RuntimeError("n >= 3")
    def is_inside(pos):
        x, y = pos
        y=y-0.289 # for the armchair edge
        x=x-1/2 # for the armchair edge
        angle = atan(y, x) - start
        angle %= (2*np.pi/n)
        alpha = (n - 2)/2/n*np.pi
        return np.sqrt(x*x + y*y) < np.sin(alpha)/np.sin(alpha + angle)*r
    return is_inside


# Failed attempt to calculate conductivity using the Kubo formula
#  =========================================================================================================
# def caclculate_conductivity(sys: kwant.builder.FiniteSystem, param: SimpleNamespace, eta: float = 1e-12, E: float = 0.0) -> float:
#     size = len(sys.sites)
#     h = sys.hamiltonian_submatrix(params=dict(param=param), sparse=False)
#     eigs, states = np.linalg.eigh(h)
#     states = states.T
#     Vx = np.zeros_like(h)
#     Vy = np.zeros_like(h)
#     sites = list(sys.sites)
#     for i, j in sys.graph:
#         Vx[i, j] = 1j*(sites[i].pos[0] - sites[j].pos[0]) * h[i, j]
#         Vy[i, j] = 1j*(sites[i].pos[1] - sites[j].pos[1]) * h[i, j]
#     if not np.allclose(Vx, Vx.T.conj()):
#         Vx += Vx.T.conj()
#         Vy += Vy.T.conj()
    
#     cond_xx = 0.0 + 0.0j
#     cond_xy = 0.0 + 0.0j
#     for i in range(size):
#         for j in range(size):
#             if i == j: continue
#             if eigs[i] * eigs[j] > 1e-12: continue
#             numerator_xx = np.abs(np.vdot(states[i], Vx @ states[j]))**2
#             numerator_xy = np.vdot(states[i], Vy @ states[j])*np.vdot(states[j], Vx @ states[i])
#             denominator = (eigs[i] - eigs[j])**2 + eta*eta
#             cond_xx += numerator_xx / denominator
#             cond_xy += numerator_xy / denominator

#     return cond_xx, cond_xy
# =========================================================================================================


def onsite(site, param):
    staggered_pot =  param.m if site.family == a_lat else -param.m
    disorder = param.disorder*kwant.digest.gauss(site.tag, "my_disorder")
    return disorder + staggered_pot
def hopping(site1, site2, param):
    x1, y1 = site1.pos
    x2, y2 = site2.pos
    return 1*np.exp(-1j*param.phi/2*(x2 - x1)*(y1 + y2))
def nnn_hopping(site1, site2, param):
    x1, y1 = site1.pos
    x2, y2 = site2.pos
    return (np.real(param.t_2) + 1j*np.imag(param.t_2))*np.exp(-1j*param.phi/2*(x2 - x1)*(y1 + y2))

def make_system(a, shape="ribbon", diminsions=SimpleNamespace(L=10, W=10)):#, phi=0.0, t_2=0.0, m=0.0):

    sys = kwant.Builder()
    if shape == "ribbon":
        for i in range(diminsions.L):
            for j in range(diminsions.W):
                sys[a_lat(i, j)] = onsite
                sys[b_lat(i, j)] = onsite
    elif shape == "rectangle":
        sys[lat.shape(lambda pos: 0<=pos[0]<=diminsions.L and 0<=pos[1]<=diminsions.W, (0, 0))] = onsite
    
    elif shape == "triangular-zz":
        for i in range(diminsions.N_edge):
            for j in range(diminsions.N_edge- i):
                sys[a_lat(i, j)] = onsite
                sys[b_lat(i, j)] = onsite
    
    elif shape == "triangular-ac":
        # sys[lat.shape(Regular_Polygon(diminsions.r, n=3, start=0), start=(0, 0))] = onsite
        def armchair_triangle(pos):
            x, y = pos
            y+=1/np.sqrt(3) 
            
            le = x > 0
            be = y >= x/np.sqrt(3) + 0.01
            re = x <= diminsions.l*np.sqrt(3)/2
            ue = y <= diminsions.l - x/np.sqrt(3)
            return le and be and re and ue
    
        sys[lat.shape(armchair_triangle, start=(1, 1))] = onsite

    # hexagon
    elif shape == "hexagon-zz":
        for i in range(0, 2*diminsions.Lx):
            if i < diminsions.Lx:
                for j in range(0, diminsions.Lx + 1):
                    sys[a_lat(i, j)] = onsite
                    sys[b_lat(i, j)] = onsite
                    if j > 0:
                        sys[a_lat(i+j, -j)] = onsite
                        sys[b_lat(i+j, -j)] = onsite
            else:
                for j in range(0, diminsions.Lx - (i - diminsions.Lx)):
                    sys[a_lat(i, j)] = onsite
                    sys[b_lat(i, j)] = onsite
                    if j > 0:
                        sys[a_lat(i+j, -j)] = onsite
                        sys[b_lat(i+j, -j)] = onsite
    
    elif shape == "hexagon-ac":
        sys[lat.shape(Regular_Polygon(diminsions.r, n=6, start=np.pi/6), start=(0, 0))] = onsite

    elif shape == "circle":
        sys[lat.shape(lambda pos: np.linalg.norm(pos) < diminsions.r, start=(0, 0))] = onsite
    
    
    sys[lat.neighbors(1)] = hopping

    for __ in range(2):
        sites = list(sys.sites())
        for s in sites:
            num_connections = sum(1 for _ in sys.neighbors(s))
            if num_connections == 1 or num_connections == 0: del sys[s]

    sys[lat_neighbors_2] = nnn_hopping

    return sys

# chatGPT
def find_edge_sites(syst: kwant.Builder, cutoff=0.9):
    """
    Returns sites with fewer than 3 first-nearest neighbors, using a distance cutoff.
    
    Parameters:
    - fsyst: Finalized Kwant system.
    - cutoff: Distance threshold to distinguish first neighbors.

    Returns:
    - List of kwant Site objects that are edge sites.
    """
    edge_sites = []
    for i, site in enumerate(syst.sites()):
        r0 = site.pos
        first_neighbors = 0
        for j, other_site in enumerate(syst.neighbors(site)):
            if np.allclose(other_site.pos, site.pos): continue
            r1 = other_site.pos
            dist = np.linalg.norm(r0 - r1)
            if dist < cutoff:
                first_neighbors += 1
        if first_neighbors < 3:
            edge_sites.append(site)
    return edge_sites


def file_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape",          type=str,       default="triangular-zz",choices=["ribbon", "rectangle", "triangular-zz", "triangular-ac", "hexagon-zz", "hexagon-ac", "circle"])
    parser.add_argument("--N_edge",         type=int,       default=25,                 help="Number of edges for triangular shapes")
    parser.add_argument("--r",              type=float,     default=11,                 help="Radius for circular and hexagonal shapes")
    parser.add_argument("--l",              type=float,     default=24,                 help="Length for triangular and hexagonal shapes")
    parser.add_argument("--Lx",             type=int,       default=15,                 help="Length in x direction for hexagonal shapes")
    parser.add_argument("--L",              type=float,     default=30.0,               help="Length in y direction for rectangular shapes")
    parser.add_argument("--W",              type=float,     default=20.0,               help="Width for rectangular shapes")
    parser.add_argument("--phi",            type=float,     default=0.0,                help="Phase factor for hopping")
    parser.add_argument("--t_2",            type=float,     default=1e-12,              help="NNN hopping parameter")
    parser.add_argument("--t_prime",        type=float,     default=0.0,                help="NNN hopping imaginary parameter")
    parser.add_argument("--m",              type=float,     default=0.0,                help="Staggered potential")
    parser.add_argument("--disorder",       type=float,     default=0.1,                help="Disorder strength")
    parser.add_argument("--num_moments",    type=int,       default=50,                 help="Number of moments for KPM")
    parser.add_argument("--energy",        type=float,     default=0.0,                 help="Energy for KPM local spectral density plots")
    return parser.parse_args()



if __name__ == "__main__":

    shapes = ["triangular-zz", "ribbon", "triangular-ac", "hexagon-zz", "hexagon-ac", "circle", "rectangle"]
    
    args = file_input()

    parameters = SimpleNamespace(phi=args.phi, t_2=args.t_2+1j*args.t_prime, m=args.m, disorder=args.disorder)
    # shape = shapes[0]
    
    shape = args.shape    

    diminsions = SimpleNamespace(r = args.r, l = args.l, N_edge = args.N_edge, Lx = args.Lx, L=args.L, W=args.W)
    sys = make_system(1.0, shape, diminsions)
    final_sys = sys.finalized()

    density_op = kwant.operator.Density(final_sys, sum=False)
    dos = kwant.kpm.SpectralDensity(final_sys, operator=density_op, num_moments=args.num_moments, params=dict(param=parameters))
    kwant.plotter.map(final_sys, dos(energy=0.0), colorbar=True)

    # eigs, states = np.linalg.eigh(final_sys.hamiltonian_submatrix(params=dict(param=parameters), sparse=False))

    c = np.array([0.0, 0.0])
    for s in final_sys.sites:
        c = c + s.pos
    c /= len(final_sys.sites)

    edge_sites = find_edge_sites(sys, cutoff=0.9)
    selected_sites = np.random.choice(len(edge_sites), size=min(6, len(edge_sites)), replace=False)
    def in_selected_sites(site):
        return site in [edge_sites[i] for i in selected_sites]

    # where = lambda site: np.linalg.norm(site.pos - c) < 1 or in_selected_sites(site)
    where = lambda site: np.linalg.norm(site.pos - c) < 1
    # where = lambda site: in_selected_sites(site)

    # component 'xx'
    s_factory = kwant.kpm.LocalVectors(final_sys, where)
    print(len(s_factory))
    cond_xx = kwant.kpm.conductivity(final_sys, alpha='x', beta='x', mean=True, 
                                    num_vectors=None, vector_factory=s_factory, params=dict(param=parameters), num_moments=args.num_moments)
    # component 'xy'
    s_factory = kwant.kpm.LocalVectors(final_sys, where)
    cond_xy = kwant.kpm.conductivity(final_sys, alpha='x', beta='y', mean=True,
                                    num_vectors=None, vector_factory=s_factory, params=dict(param=parameters), num_moments=args.num_moments)
    energies = cond_xx.energies
    cond_array_xx = np.abs(np.array([cond_xx(e, temperature=0.0) for e in energies]))
    cond_array_xy = np.abs(np.array([cond_xy(e, temperature=0.0) for e in energies]))

    area_per_site = np.abs(np.cross(*lat.prim_vecs)) / len(lat.sublattices)
    cond_array_xx /= area_per_site
    cond_array_xy /= area_per_site
    plt.figure(figsize=(10, 5))
    plt.plot(energies, cond_array_xx, label=r'$\sigma_{xx}$', color='blue')
    plt.plot(energies, cond_array_xy, label=r'$\sigma_{xy}$', color='red')
    plt.plot(energies, np.sqrt(cond_array_xx**2 + cond_array_xy**2), label=r'$\sigma$', color='green')
    plt.xlabel('Energy')
    plt.ylabel('Conductivity (e^2/h)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
