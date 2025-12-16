import numpy as np
import matplotlib.pyplot as plt
import kwant
import scipy
import tinyarray
from scipy import linalg as la
from scipy.sparse import linalg as sla
from types import SimpleNamespace
import argparse
from copy import deepcopy
from utils2 import Regular_Polygon
import matplotlib.cm as cm
import matplotlib.colors as colors
import timeit

lat = kwant.lattice.honeycomb(a=1.0, norbs=1)
a_lat, b_lat = lat.sublattices

nnn_hoppings_a = (((-1, 0), a_lat, a_lat), ((0, 1), a_lat, a_lat), ((1, -1), a_lat, a_lat))
nnn_hoppings_b = (((1, 0), b_lat, b_lat), ((0, -1), b_lat, b_lat), ((-1, 1), b_lat, b_lat))
nnn_hoppings_all = nnn_hoppings_a + nnn_hoppings_b
lat_neighbors_2 = [kwant.builder.HoppingKind(*hop) for hop in nnn_hoppings_all]

SHAPES = ["triangular-zz", "ribbon", "triangular-ac", "hexagon-zz", "hexagon-ac", "circle", "rectangle"]


def make_system(shape=SHAPES[-1], diminsions=SimpleNamespace(L=10, W=10)):

    def onsite(site, param):
        staggered_pot =  param.m if site.family == a_lat else -param.m
        disorder = param.disorder*kwant.digest.gauss(repr(site), "")
        return (disorder + staggered_pot)
    
    def hopping(site1, site2, param):
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        return -1
    
    def nnn_hopping(site1, site2, param):
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        t_nnn = np.real(param.t_2) + 1j*np.imag(param.t_2)
        return t_nnn

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
        sys[lat.shape(Regular_Polygon(diminsions.R, n=6, start=np.pi/6), start=(0, 0))] = onsite
    elif shape == "circle":
        sys[lat.shape(lambda pos: np.linalg.norm(pos) < diminsions.r, start=(0, 0))] = onsite
    else:
        raise ValueError(f"Shape {shape} not recognized")
    
    sys[lat.neighbors()] = hopping
    edges = []

    for __ in range(2):
        sites = list(sys.sites())
        for s in sites:
            num_connections = sum(1 for _ in sys.neighbors(s))
            if num_connections == 1 or num_connections == 0: del sys[s]
            elif num_connections ==2: edges.append(s)
            else: pass

    sys[lat_neighbors_2] = nnn_hopping

    return sys, edges

# attach leads
def add_leads(sys, shape, diminsions, add_second_lead=True):
    sys_with_leads = deepcopy(sys)
    sq_lattice = kwant.lattice.square(a=1.0, norbs=1)
    add_second_lead = True

    if shape == "triangular-zz":    
        # triangle -zz
        added_site =  sq_lattice(0, 1)
        _lead1 = kwant.Builder(kwant.TranslationalSymmetry([-1, 0])) # w a(0, 1)
        added_site2 = sq_lattice(diminsions.N_edge-1, 1)
        _lead2 = kwant.Builder(kwant.TranslationalSymmetry([1, 0])) # w b(10, 0)
    elif shape == "triangular-ac":
        # triangle -ac
        added_site = sq_lattice(0, diminsions.l/2)
        _lead1 = kwant.Builder(kwant.TranslationalSymmetry([-1, 0])) # w a(0, 1)
        added_site2 = sq_lattice(diminsions.l, diminsions.l/2)
        _lead2 = kwant.Builder(kwant.TranslationalSymmetry([1, 0])) # w b(5, 5)
    elif shape == "hexagon-zz":
        # hexagon -zz
        added_site = sq_lattice(-1, 0)
        _lead1 = kwant.Builder(kwant.TranslationalSymmetry([-1, 0])) # w a(0, 0)
        added_site2 = sq_lattice(2*diminsions.Lx, 0)
        _lead2 = kwant.Builder(kwant.TranslationalSymmetry([1, 0])) # w b(29, 0)
    elif shape == "hexagon-ac":
    # hexagon -ac
        added_site = sq_lattice(-diminsions.R, 0)
        _lead1 = kwant.Builder(kwant.TranslationalSymmetry([-1, 0])) # w a(-12, 0)
        added_site2 = sq_lattice(diminsions.R, 0)
        _lead2 = kwant.Builder(kwant.TranslationalSymmetry([1, 0])) # w b(12, 0)
    elif shape == "circle":
    # circle
        added_site = sq_lattice(-diminsions.r, 0)
        _lead1 = kwant.Builder(kwant.TranslationalSymmetry([-1, 0])) # w a(-6, 0)
        added_site2 = sq_lattice(diminsions.r, 0)
        _lead2 = kwant.Builder(kwant.TranslationalSymmetry([1, 0])) # w  a(6, 0)

    elif shape == "rectangle":
        # rectangle
        added_site = sq_lattice(0, 0)
        _lead1 = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
        
        added_site2 = sq_lattice(diminsions.L, diminsions.W)
        _lead2 = kwant.Builder(kwant.TranslationalSymmetry([0, 1]))
            
    else:
        raise ValueError(f"Shape {shape} not recognized for lead attachment")


    w1 = list(sys.sites())[0]
    w2 = list(sys.sites())[-1]
    for s in sys.sites():
        d1 = np.linalg.norm(w1.pos - added_site.pos)
        d2 = np.linalg.norm(w2.pos - added_site2.pos)
        if np.linalg.norm(s.pos - added_site.pos) < d1:
            w1 = s
        if np.linalg.norm(s.pos - added_site2.pos) < d2:
            w2 = s

    _lead1[added_site] = 0
    _lead1[sq_lattice.neighbors(1)] = -3
    sys_with_leads[added_site]  = 0
    sys_with_leads[added_site,  w1] = -3
    sys_with_leads.attach_lead(_lead1)

    if add_second_lead:
        _lead2[added_site2] = 0
        _lead2[sq_lattice.neighbors(1)] = -3
        sys_with_leads[added_site2] = 0
        sys_with_leads[added_site2, w2] = -3
        sys_with_leads.attach_lead(_lead2)


    sys_with_leads_fin = sys_with_leads.finalized()
    return sys_with_leads_fin


def psi_up_dn(sys_fin, energy, params, edges_up, edges_down):
    psi_r = kwant.wave_function(sys_fin, energy=energy, params=params)(0)[0]
    psi_l = kwant.wave_function(sys_fin, energy=energy, params=params)(1)[0]
    if edges_up is not None and edges_down is not None:
        density_up = kwant.operator.Density(sys_fin, sum=True, where=lambda site: site in edges_up)(psi_r)/len(edges_up) # divided by number of edges to get the average, though it was necessary for the triangular-zz shape
        density_down = kwant.operator.Density(sys_fin, sum=True, where=lambda site: site in edges_down)(psi_r)/len(edges_down)
        psi_up_l = kwant.operator.Density(sys_fin, sum=True, where=lambda site: site in edges_up)(psi_l)/len(edges_up)
        psi_down_l = kwant.operator.Density(sys_fin, sum=True, where=lambda site: site in edges_down)(psi_l)/len(edges_down)
    else:
        density_up = kwant.operator.Density(sys_fin, sum=True, where=lambda site: site.pos[1] > site.pos[0])(psi_r)
        density_down = kwant.operator.Density(sys_fin, sum=True, where=lambda site: site.pos[0] > site.pos[1])(psi_r)
        psi_up_l = kwant.operator.Density(sys_fin, sum=True, where=lambda site:  site.pos[1] > site.pos[0])(psi_l)
        psi_down_l = kwant.operator.Density(sys_fin, sum=True, where=lambda site:  site.pos[0] > site.pos[1])(psi_l)
    return density_up, density_down, psi_up_l, psi_down_l

def file_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape",          type=str,       default="triangular-zz",        choices=["ribbon", "rectangle", "triangular-zz", "triangular-ac", "hexagon-zz", "hexagon-ac", "circle"])
    parser.add_argument("--N_edge",         type=int,       default=25,                     help="Number of edges for triangular shapes")
    parser.add_argument("--r",              type=float,     default=11,                     help="Radius for circular and hexagonal shapes")
    parser.add_argument("--R",              type=float,     default=9.5,                    help="Radius for circular and hexagonal shapes")
    parser.add_argument("--l",              type=float,     default=24,                     help="Length for triangular and hexagonal shapes")
    parser.add_argument("--Lx",             type=int,       default=15,                     help="Length in x direction for hexagonal shapes")
    parser.add_argument("--L",              type=float,     default=30.0,                   help="Length in y direction for rectangular shapes")
    parser.add_argument("--W",              type=float,     default=20.0,                   help="Width for rectangular shapes")
    parser.add_argument("--phi",            type=float,     default=0.0,                    help="Phase factor for hopping")
    parser.add_argument("--t_2",            type=float,     default=1e-12,                  help="NNN hopping parameter")
    parser.add_argument("--t_prime",        type=float,     default=0.0,                    help="NNN hopping imaginary parameter")
    parser.add_argument("--m",              type=float,     default=0.0,                    help="Staggered potential")
    parser.add_argument("--disorder",       type=float,     default=0.1,                    help="Disorder strength")
    parser.add_argument("--energy",         type=float,     default=0.0,                    help="Energy for KPM local spectral density plots")
    parser.add_argument("--plot",           type=bool,      default=True,                  help="Plot dot and energies")
    parser.add_argument("--plot_densities",           type=bool,      default=False,                  help="Plot local densities and currents")
    parser.add_argument("--E0",           type=float,      default=-1.000,                  help="starting energy")
    parser.add_argument("--E1",           type=float,      default=1.001,                  help="last energy energy")
    parser.add_argument("--N_e",           type=int,      default=251,                  help="energy points")
    return parser.parse_args()


if __name__ == "__main__":
    
    
    args = file_input()
    
    # Define the shape of the system
    shapes = ["triangular-zz", "ribbon", "triangular-ac", "hexagon-zz", "hexagon-ac", "circle", "rectangle"]
    shape = args.shape
    if shape not in shapes:
        raise ValueError(f"Shape {shape} not recognized. Choose from {shapes}")

    # Define the dimensions of the system

    diminsions = SimpleNamespace(r=args.r, R=args.R, l=args.l, Lx=args.Lx, L=args.L, W=args.W, N_edge=args.N_edge)
    # diminsions.N_edge = 50 # will have perfect transmission that descends slowly, but ascends later that I aim for

    # Create the system
    sys, edges = make_system(shape, diminsions)

    edges_up = [s for s in edges if s.pos[1] > 0]
    edges_down = [s for s in edges if s.pos[1] <= 0]

    # define special edges
    if shape == "triangular-zz":
        min_y = min(s.pos[1] for s in edges_up)
        edges_up = []
        edges_down = []
        for s in edges:
            if np.abs(s.pos[1] - min_y)<0.5: edges_down.append(s)
            else: edges_up.append(s)
    elif shape == "rectangle":
        edges_up = []
        edges_down = []
        # for s in edges: # I changed this to consider all sites, not just edges, which doesn't make much sense
        for s in sys.sites():
            if s.pos[1]*args.L > s.pos[0]*args.W: edges_up.append(s)
            else: edges_down.append(s)
    elif shape == "triangular-ac":
        print("Not yet handled")
        edges_up = []
        edges_down = []
        for s in edges:
            if s.pos[1] > diminsions.l/2: edges_up.append(s)
            else: edges_down.append(s)


    system_parameters = SimpleNamespace(t_2=args.t_2 + 1j * args.t_prime, m=args.m, disorder=args.disorder)


    sys_with_leads_fin = add_leads(sys, shape, diminsions)

    def color_edges(site):
        site = sys_with_leads_fin.sites[site]
        if site in edges_up:
            return 'orange'
        elif site in edges_down:
            return 'red'
        else:
            return 'blue'
    kwant.plot(sys_with_leads_fin, site_size=0.25, show=False, site_color=color_edges)
    # kwant.plot(sys_with_leads_fin, site_size=0.25, show=True)
    if shape == "triangular-zz": plt.hlines(min_y+0.5, -2, 12, color='red', lw=0.5)
    if args.plot: plt.show()
    
    # Plotting local densities and currents
    plot_the_densities = args.plot_densities

    if plot_the_densities:
        e = args.energy - args.m/2 - args.t_2
        wf = kwant.wave_function(sys_with_leads_fin, e, params=dict(param=system_parameters))
        try:
            psi = wf(0)[0]
            psi_inv = wf(1)[0]
        except IndexError:
            print("No wave function at the added site, check the energy.")
        rho = kwant.operator.Density(sys_with_leads_fin)
        density = rho(psi, params=dict(param=system_parameters))
        density_inv = rho(psi_inv, params=dict(param=system_parameters))

        J0 = kwant.operator.Current(sys_with_leads_fin)
        current = J0(psi, params=dict(param=system_parameters))
        current_inv = J0(psi_inv, params=dict(param=system_parameters))
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))

        # 1. Local electron density
        vmin, vmax = 0, 1
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.Reds
        kwant.plotter.map(sys_with_leads_fin, density, ax=axs[0], cmap=cmap, vmin=vmin, vmax=vmax, show=False)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axs[0])
        axs[0].set_title("Local electron density")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")

        # 3. Local current
        vmin, vmax = current.min(), current.max()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.seismic
        kwant.plotter.current(sys_with_leads_fin, current, ax=axs[1], cmap=cmap, show=False)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axs[1])
        axs[1].set_title("Local current")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        
        plt.tight_layout()
        plt.show()


        fig, axs = plt.subplots(2, figsize=(12, 10))

        zcurrent_00 = J0(wf(0)[0], params=dict(param=system_parameters))
        vmin, vmax = zcurrent_00.min(), zcurrent_00.max()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.seismic
        kwant.plotter.current(sys_with_leads_fin, zcurrent_00, ax=axs[0], cmap=cmap, show=False)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axs[0])
        axs[0].set_title("Mode 0 Lead 0 Current")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")

        zcurrent_01 = J0(wf(1)[0], params=dict(param=system_parameters))
        vmin, vmax = zcurrent_01.min(), zcurrent_01.max()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.seismic
        kwant.plotter.current(sys_with_leads_fin, zcurrent_01, ax=axs[1], cmap=cmap, show=False)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axs[1])
        axs[1].set_title("Mode 0 Lead 1 Current")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")

        plt.tight_layout()
        plt.show()

    start_time = timeit.default_timer()

    energies = np.linspace(args.E0, args.E1, args.N_e)
    transmissions = []
    conductance = []
    for energy in energies:
        smatrix = kwant.smatrix(sys_with_leads_fin, energy, params=dict(param=system_parameters))
        T = smatrix.transmission(0, 1)
        transmissions.append(T)
        u_r, d_r, u_l, d_l = psi_up_dn(sys_with_leads_fin, energy, dict(param=system_parameters), edges_down=edges_down,edges_up=edges_up)
        conductance.append(T * np.abs((u_r + u_l)*(d_r + d_l)/(u_l*d_r - u_r*d_l)))


    # t_primes = 0.001*(1/0.001)**(np.array(range(201))/200)  # 0.001 to 1.0 in 100 steps
    # transmissions_tprime = []
    # conductance_tprime = []

    # for t_prime_ in t_primes:
    #     params = dict(param=SimpleNamespace(t_2=1j * t_prime_ + args.t_2, m=args.m, disorder=args.disorder))
    #     smatrix = kwant.smatrix(sys_with_leads_fin, 0.0, params=params)
    #     T = smatrix.transmission(0, 1)
    #     transmissions_tprime.append(T)
    #     # u_r, d_r, u_l, d_l = psi_up_dn(sys_with_leads_fin, 0.0, params, None, None)
    #     u_r, d_r, u_l, d_l = psi_up_dn(sys_with_leads_fin, 0.0, params, edges_up=edges_up, edges_down=edges_down)
    #     conductance_tprime.append(T * np.abs((u_r + u_l)*(d_r + d_l)/(u_l*d_r - u_r*d_l)))

    print(f"Total time taken: {timeit.default_timer() - start_time:.2f} seconds")    

    plt.figure(figsize=(7, 5))
    plt.plot(energies, transmissions, label="T(0→1)")
    plt.xlabel("Energy")
    plt.ylabel("Transmission coefficient")
    plt.legend()
    plt.grid(True)
    if args.shape == "circle":
        np.save(f"trans--r-{args.r}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.npy", transmissions)
        plt.savefig(f"trans--r-{args.r}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.pdf")
    elif args.shape == "hexagon-ac":
        np.save(f"trans--R-{args.R}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.npy", transmissions)
        plt.savefig(f"trans--R-{args.R}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.pdf")
    elif args.shape == "hexagon-zz":
        np.save(f"trans--Lx-{args.Lx}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.npy")
        plt.savefig(f"trans--Lx-{args.Lx}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.pdf")
    elif args.shape == "triangular-ac":
        np.save(f"trans--l-{args.l}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.npy", transmissions)
        plt.savefig(f"trans--l-{args.l}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.pdf")
    elif args.shape == "triangular-zz":
        np.save(f"trans--N_edge-{args.N_edge}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.npy", transmissions)
        plt.savefig(f"trans--N_edge-{args.N_edge}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.pdf")
    elif args.shape == "rectangle":
        np.save(f"trans--L-{args.L}--W-{args.W}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.npy", transmissions)
        plt.savefig(f"trans--L-{args.L}--W-{args.W}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.pdf")
    else:
        print(f"Shape {args.shape} not recognized for saving transmission plot")
    plt.tight_layout()
    plt.grid(True)
    if args.plot: plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(energies, conductance, label="Conductance", color='orange')
    plt.xlabel("Energy")
    plt.ylabel(r"Conductance $\left[\frac{e^2}{h}\right]$")
    plt.ylim(0, 5.0)
    plt.legend()
    plt.grid(True)
    if args.shape == "circle":
        np.save(f"cond--r-{args.r}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.npy", conductance)
        plt.savefig(f"cond--r-{args.r}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.pdf")
    elif args.shape == "hexagon-ac":
        np.save(f"cond--R-{args.R}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.npy", conductance)
        plt.savefig(f"cond--R-{args.R}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.pdf")
    elif args.shape == "hexagon-zz":
        np.save(f"cond--Lx-{args.Lx}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.npy", conductance)
        plt.savefig(f"cond--Lx-{args.Lx}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.pdf")
    elif args.shape == "triangular-ac":
        np.save(f"cond--l-{args.l}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.npy", conductance)
        plt.savefig(f"cond--l-{args.l}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.pdf")
    elif args.shape == "triangular-zz":
        np.save(f"cond--N_edge-{args.N_edge}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.npy", conductance)
        plt.savefig(f"cond--N_edge-{args.N_edge}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.pdf")
    elif args.shape == "rectangle":
        np.save(f"cond--L-{args.L}--W-{args.W}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.npy", conductance)
        plt.savefig(f"cond--L-{args.L}--W-{args.W}--t_prime-{args.t_prime}--t_2-{args.t_2}--m-{args.m}--disorder-{args.disorder}.pdf")
    else:
        print(f"Shape {args.shape} not recognized for saving transmission plot")
    plt.tight_layout()
    if args.plot: plt.show()


    # plt.figure(figsize=(7, 5))
    # plt.plot(t_primes, transmissions_tprime, label="T(0→1)")
    # plt.xlabel(r"SOC Strength (t')")
    # plt.ylabel("Transmission coefficient")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # plt.figure(figsize=(7, 5))
    # plt.plot(t_primes, conductance_tprime, label="Conductance")
    # # plt.plot(t_primes[645:], conductance[645:], label="Conductance")
    # plt.xlabel(r"SOC Strength (t')", fontsize=10)
    # plt.ylabel(r"Conductance\left[\frac{e^2}{h}\right]")
    # plt.ylim(0, 5)
    # plt.legend()
    # plt.grid(True)
    # plt.show()

