import numpy as np
import matplotlib.pyplot as plt
import kwant
import tinyarray
import scipy
from scipy import linalg as la
from scipy.sparse import linalg as sla
from matplotlib.patches import ConnectionPatch
from types import SimpleNamespace
from copy import deepcopy
from utils2 import binary_search, calculate_participation_ratio

sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])
sigma_0 = tinyarray.array([[1, 0], [0, 1]])

# lat = kwant.lattice.honeycomb(a=1.0, norbs=2)
# a_lat, b_lat = lat.sublattices
lat = kwant.lattice.general([[1, 0], [1/2, np.sqrt(3)/2]],  # lattice vectors
                      [[0, 0], [0, 1/np.sqrt(3)]], norbs=2)  # Coordinates of the sites
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

def onsite(site, param):
    staggered_pot =  param.u_s if site.family == a_lat else -param.u_s
    disorder = param.dicorder*kwant.digest.gauss(repr(site), "")
    return disorder*sigma_0 + staggered_pot*sigma_z
def hopping(site1, site2, param):
    x1, y1 = site1.pos
    x2, y2 = site2.pos
    return -1*np.exp(-1j*param.phi/2*(x2 - x1)*(y1 + y2))*sigma_0

def nnn_hopping(site1, site2, param):
    x1, y1 = site1.pos
    x2, y2 = site2.pos
    return -(np.real(param.t_2)*sigma_0 + 1j*np.imag(param.t_2)*sigma_z)*np.exp(-1j*param.phi/2*(x2 - x1)*(y1 + y2))



def make_system(a, shape="ribbon", diminsions=SimpleNamespace(L=10, W=10)):#, phi=0.0, t_2=0.0, u_s=0.0):

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
 

def edge_prob(syst: kwant.Builder, eig_state: np.ndarray) -> float:
    edge_prob = 0.0
    sites = list(syst.sites())
    for i, s in enumerate(sites):
        num_connections = sum(1 for _ in sys.neighbors(s))
        
        if num_connections <9:
            edge_prob= edge_prob + (np.abs(eig_state[2*i])**2 + np.abs(eig_state[2*i+1])**2)
    return edge_prob/np.linalg.norm(eig_state)**2


if __name__ == "__main__":
    # Define the lattice constant
    a = 1.0
    add_second_lead = True
    # Define the shape of the system
    shapes = ["triangular-zz", "ribbon", "triangular-ac", "hexagon-zz", "hexagon-ac", "circle", "rectangle"]
    parameters = SimpleNamespace(phi=0.0, t_2=1e-12+0.2j, u_s=0.0, dicorder=0.0)
    shape = shapes[0]
    
    # Define the dimensions of the system
    diminsions = SimpleNamespace(r=11, l=17, N_edge=10, Lx = 10, L=15, W=10)
    # diminsions.N_edge = 30

    # Create the system
    sys = make_system(a, shape, diminsions)
    # kwant.plot(sys)

    # Isolated system, wave function and density, and local spin polarized currents
    # ========================================================================================================
    
    final_sys = sys.finalized()
    # compare eigenvalues with and without spin orbit coupling
    parameters.t_2 = 1e-12 + 0.0787874j #0.05j
    eigs_soc, states_soc = np.linalg.eigh(final_sys.hamiltonian_submatrix(params=dict(param=parameters), sparse=False))
    states_soc = states_soc.T
    parameters.t_2 = 1e-12
    eigs_nosoc, states_nosoc = np.linalg.eigh(final_sys.hamiltonian_submatrix(params=dict(param=parameters), sparse=False))
    states_nosoc = states_nosoc.T

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (0, 1))
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    ax1.plot(eigs_nosoc, 'o', label="No SOC")
    ax1.plot(eigs_soc, 'o', label="SOC")
    # ax1.set_xlabel('Index')
    ax1.set_ylabel('Energy')
    ax1.grid()

    print(edge_prob(sys, states_soc[90]))

    kwant.plot(sys, show=False, ax=ax2, hop_lw=0.01, fig_size=(5, 2.5))
    index = 92#binary_search(eigs_soc, 0.0)
    image = np.abs(states_soc[index][0::2])**2# + np.abs(states_soc[index][1::2])**2
    image_nosoc = np.abs(states_nosoc[index][0::2])**2# + np.abs(states_nosoc[index][1::2])**2
    kwant.plotter.map(final_sys, image/image.max(), colorbar=False, show=True, oversampling=10, vmax=1.00, ax=ax3)
    ax3.set_title("SOC")
    kwant.plotter.map(final_sys, image_nosoc/image_nosoc.max(), colorbar=False, show=True, oversampling=10, vmax=1.00, ax=ax4)
    ax4.set_title("No SOC")
    plt.show()


    # Plotting the zeroenergy eigenstates
    # ========================================================================================================
    ax1 = plt.subplot2grid((1, 7), (0, 0))
    ax2 = plt.subplot2grid((1, 7), (0, 1)); ax2.set_yticklabels([])
    ax3 = plt.subplot2grid((1, 7), (0, 2)); ax3.set_yticklabels([])
    ax4 = plt.subplot2grid((1, 7), (0, 3)); ax4.set_yticklabels([])
    ax5 = plt.subplot2grid((1, 7), (0, 4)); ax5.set_yticklabels([])
    ax6 = plt.subplot2grid((1, 7), (0, 5)); ax6.set_yticklabels([])
    ax7 = plt.subplot2grid((1, 7), (0, 6)); ax7.set_yticklabels([])

    index = 90
    image = np.abs(states_soc[index][0::2])**2 + np.abs(states_soc[index][1::2])**2
    kwant.plotter.map(final_sys, image/image.max(), colorbar=False, show=True, oversampling=10, vmax=1.00, ax=ax1)
    index = 92
    image = np.abs(states_soc[index][0::2])**2 + np.abs(states_soc[index][1::2])**2
    kwant.plotter.map(final_sys, image/image.max(), colorbar=False, show=True, oversampling=10, vmax=1.00, ax=ax2)
    index = 94
    image = np.abs(states_soc[index][0::2])**2 + np.abs(states_soc[index][1::2])**2
    kwant.plotter.map(final_sys, image/image.max(), colorbar=False, show=True, oversampling=10, vmax=1.00, ax=ax3)
    index = 96
    image = np.abs(states_soc[index][0::2])**2 + np.abs(states_soc[index][1::2])**2
    kwant.plotter.map(final_sys, image/image.max(), colorbar=False, show=True, oversampling=10, vmax=1.00, ax=ax4)
    index = 98
    image = np.abs(states_soc[index][0::2])**2 + np.abs(states_soc[index][1::2])**2
    kwant.plotter.map(final_sys, image/image.max(), colorbar=False, show=True, oversampling=10, vmax=1.00, ax=ax5)
    index = 100
    image = np.abs(states_soc[index][0::2])**2 + np.abs(states_soc[index][1::2])**2
    kwant.plotter.map(final_sys, image/image.max(), colorbar=False, show=True, oversampling=10, vmax=1.00, ax=ax6)
    index = 102
    image = np.abs(states_soc[index][0::2])**2 + np.abs(states_soc[index][1::2])**2
    kwant.plotter.map(final_sys, image/image.max(), colorbar=False, show=True, oversampling=10, vmax=1.00, ax=ax7)
    plt.show()

    

    PRs = []
    soc_coeffs = np.linspace(0.0, 0.5, 100)
    for t_so in soc_coeffs:
        PRs_ = []
        for s in range(90, 104, 2):
            parameters.t_2 = 1e-12 + 1j*t_so
            _, states_soc = np.linalg.eigh(final_sys.hamiltonian_submatrix(params=dict(param=parameters), sparse=False))
            states_soc = states_soc.T
            PRs_.append(edge_prob(sys, states_soc[s]))
            # PRs_.append(calculate_participation_ratio(states_soc[s]))
        PRs.append(PRs_)
    PRs = np.array(PRs)
    fig, ax = plt.subplots()
    for i in range(7):
        ax.plot(soc_coeffs, PRs[:, i], label=fr"$\psi_{i+1}$")
    ax.set_xlabel("SOC coefficient")
    ax.set_ylabel("Edge probability")
    # ax.set_title("Participation Ratio vs SOC coefficient")
    ax.legend()
    ax.grid()
    plt.show()

    


    while False:
        ind = input("Enter the index of the eigenstate to plot (or -1 to exit): ")
        if ind.strip() == "-1":
            break
        elif ind.strip().capitalize() == "E":
            fig, ax = plt.subplots()
            ax.plot(eigs_nosoc, 'o', label="No SOC")
            ax.plot(eigs_soc, 'o', label="SOC")
            ax.set_xlabel('Index')
            ax.set_ylabel('Energy')
            ax.set_title('Eigenenergies')
            ax.grid()
            plt.show()
        else:
            ind = int(ind.strip())
            state = states_soc[ind]
            ax1 = plt.subplot2grid((1, 2), (0, 0))
            ax2 = plt.subplot2grid((1, 2), (0, 1))
            kwant.plotter.map(final_sys, np.abs(state[0::2])**2 + np.abs(state[1::2])**2, colorbar=True, show=False, oversampling=10, ax=ax1)
            ax1.set_title("SOC")
            kwant.plotter.map(final_sys, np.abs(states_nosoc[ind][0::2])**2 + np.abs(states_nosoc[ind][1::2])**2, colorbar=True, show=False, oversampling=10, ax=ax2)
            ax2.set_title("No SOC")
            plt.show()


    # Leads, conductivity and scattering wave function (connected systems)
    # =======================================================================================================
    # sys_with_leads = deepcopy(sys)

    # # attach leads
    # sq_lattice = kwant.lattice.square(a=1.0, norbs=1)
    # added_site =  sq_lattice(0, 1)

    # _lead1 = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
    # _lead1[added_site] = 0
    # _lead1[sq_lattice.neighbors(1)] = -1


    # sys_with_leads[added_site]  = 0
    # sys_with_leads[added_site,  b_lat(0, 1)] = -1

    # sys_with_leads.attach_lead(_lead1)

    # if add_second_lead:
    #     added_site2 = sq_lattice(diminsions.N_edge-2, 0)
    #     _lead2 = kwant.Builder(kwant.TranslationalSymmetry([0, -1]))
    #     _lead2[added_site2] = 0
    #     _lead2[sq_lattice.neighbors(1)] = -1
    #     sys_with_leads[added_site2] = 0
    #     sys_with_leads[added_site2, b_lat(diminsions.N_edge-2, 0)] = -1
    #     sys_with_leads.attach_lead(_lead2)


    # sys_with_leads_fin = sys_with_leads.finalized()

    # kwant.plot(sys_with_leads_fin, show=True)
    
    # scattering_wave_function = kwant.wave_function(sys=sys_with_leads_fin, energy=0.0, params=dict(param=parameters))
    # den = (np.abs(scattering_wave_function(0))**2).sum(axis=0)
    # kwant.plotter.map(sys_with_leads_fin, den, colorbar=True, show=True)
    # # print(np.linalg.norm(scattering_wave_function(0)[0]), scattering_wave_function(0)[0].shape)

    # # total electron current
    # current = kwant.operator.Current(sys_with_leads_fin)
    # i = current(scattering_wave_function(0)[0]/np.linalg.norm(scattering_wave_function(0)[0]), params=dict(param=parameters))
    # kwant.plotter.current(sys_with_leads_fin, i, show=True)

    # V = 10e-2
    # # energies = np.linspace(-V/2, V/2, 100)
    # socs = 0.001*(10/0.001)**(np.arange(501)/500)
    # transmissions = []

    # en = 0.0
    
    # for i in range(len(socs)):
    #     parameters.t_2 = 1e-12 + 1j*socs[i]
    #     sm = kwant.smatrix(sys_with_leads_fin, energy=en, params=dict(param=parameters))
    #     transmissions.append(sm.transmission(1, 0))
    # plt.figure()
    # plt.semilogx(socs, transmissions)
    # plt.xlabel("SOC coefficient")
    # plt.ylabel("Conductance[e^2/h]")
    # plt.grid(True, which='both')
    # plt.show()
