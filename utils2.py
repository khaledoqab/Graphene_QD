import numpy as np
import matplotlib.pyplot as plt
import kwant
import tinyarray
import scipy
from scipy import linalg as la
from scipy.sparse import linalg as sla
from matplotlib.patches import ConnectionPatch
from types import SimpleNamespace

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

lat = kwant.lattice.honeycomb(1.0, norbs=1)
a_lat, b_lat = lat.sublattices

nnn_hoppings_a = (((-1, 0), a_lat, a_lat), ((0, 1), a_lat, a_lat), ((1, -1), a_lat, a_lat))
nnn_hoppings_b = (((1, 0), b_lat, b_lat), ((0, -1), b_lat, b_lat), ((-1, 1), b_lat, b_lat))
nnn_hoppings_all = nnn_hoppings_a + nnn_hoppings_b
lat_neighbors_2 = [kwant.builder.HoppingKind(*hop) for hop in nnn_hoppings_all]

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

def plot_system(sys, ax=None):
    """
    Plot the system using matplotlib.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Get the coordinates of the sites
    coords = np.array([site.pos for site in sys.sites()])
    # Get the colors of the sites
    colors = np.array([site.family.name for site in sys.sites()])

    # Plot the sites
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=100)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    return fig, ax
def plot_hopping(sys, ax=None):
    """
    Plot the hopping terms of the system using matplotlib.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Get the coordinates of the sites
    coords = np.array([site.pos for site in sys.sites()])
    # Get the colors of the sites
    colors = np.array([site.family.name for site in sys.sites()])

    # Plot the sites
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=100)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot the hopping terms
    for site1, site2 in sys.hoppings():
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        ax.plot([x1, x2], [y1, y2], 'k-')

    plt.show()
    return fig, ax


def binary_search(arr, target):
    """
    Perform binary search on a sorted array to find the index of the target value.
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left

def plot_eigenstates(sys, eigenenergies, eigenstates, ind=None, ax=None):
    if ind is None:
        ind =  binary_search(eigenenergies, 0)
    
    fig1, ax1 = plt.subplots(figsize=[8,6.4], dpi=500)
    ax1.scatter(range(len(eigenenergies)), eigenenergies)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue (E/t)')
    ax1.axhline(0, color='red', lw=0.5)
    ax1.grid(True)

    psi_squared = np.square(np.abs(eigenstates[ind]))
    inset_ax1 = fig1.add_axes([0.15, 0.58, 0.3, 0.4])
    inset_ax1.set_yticks([])
    inset_ax1.set_xticks([])
    kwant.plotter.map(sys, psi_squared, ax=inset_ax1, colorbar=True, show=False, fig_size=(6, 4), oversampling=10)
    # plt.axis('equal')
    inset_ax1.grid(True)


    xy_main = (ind, eigenenergies[ind])
    xy_inset = (20, 8)

    # Draw arrow from inset to main axes
    con = ConnectionPatch(xyA=xy_inset, coordsA=inset_ax1.transData,
                        xyB=xy_main, coordsB=ax1.transData,
                        arrowstyle="-", color="k", lw=1)
    con2 = ConnectionPatch(xyA=(0., 0.5), coordsA=inset_ax1.transData,
                        xyB=xy_main, coordsB=ax1.transData,
                        arrowstyle="-", color="k", lw=1)

    fig1.add_artist(con)
    fig1.add_artist(con2)

    inset_ax2 = fig1.add_axes([0.60, 0.15, 0.3, 0.3])
    # inset_ax2.set_yticklabels([])
    inset_ax2.yaxis.set_ticks_position("right")
    inset_ax2.set_xticklabels([])
    plt.plot(eigenenergies[ind - 2: ind + 1], 'o')
    plt.grid(True)
    plt.savefig('./test.png', dpi=400)
    plt.show()

def plot_eigenenergies(eigenenergies, system):
    fig, ax = plt.subplots()
    ax.plot(eigenenergies, 'o')
    ax.set_xlabel('Index')
    ax.set_ylabel('Energy')
    ax.set_title('Eigenenergies')
    ax.grid()

    inset_ax1 = fig.add_axes([0.6, 0.1, 0.3, 0.45])
    inset_ax1.set_yticks([])
    inset_ax1.set_xticks([])
    kwant.plot(sys=system, ax=inset_ax1, show=False)
    plt.show()


def calculate_participation_ratio(eigenstate):
    return (np.sum(np.abs(eigenstate)**2))**2/np.sum(np.abs(eigenstate)**4)/len(eigenstate)


def get_eigenvalues(system):
    fsys = system.finalized()
    if len(fsys.sites) <= 2**12:
        h = fsys.hamiltonian_submatrix()
        eigenenergies = np.linalg.eigvalsh(h)
    else:
        h = fsys.hamiltonian_submatrix(sparse=True)
        eigenenergies = sla.eigsh(h, k=h.shape[0] - 2, which='SM', return_eigenvectors=False)[0]
        sort_arg = np.argsort(eigenenergies)
        eigenenergies = eigenenergies[sort_arg]
    return eigenenergies
