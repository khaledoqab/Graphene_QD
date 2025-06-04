import numpy as np
import matplotlib.pyplot as plt
import kwant
import scipy.sparse.linalg as sla
import scipy.sparse as sp
from scipy.constants import hbar, e, m_e, h
from types import SimpleNamespace
import tinyarray
from utils2 import binary_search, Regular_Polygon
import argparse
import time
import sys
import os

lat = kwant.lattice.general([[1, 0], [1/2, np.sqrt(3)/2]], [[0, 0], [0, 1/np.sqrt(3)]], norbs=2)
a_lat, b_lat = lat.sublattices
nnn_hoppings_a = (((-1, 0), a_lat, a_lat), ((0, 1), a_lat, a_lat), ((1, -1), a_lat, a_lat))
nnn_hoppings_b = (((1, 0), b_lat, b_lat), ((0, -1), b_lat, b_lat), ((-1, 1), b_lat, b_lat))
nnn_hoppings_all = nnn_hoppings_a + nnn_hoppings_b
lat_neighbors_2 = [kwant.builder.HoppingKind(*hop) for hop in nnn_hoppings_all]

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])


def onsite(site, param):
    staggered_pot =  param.m if site.family == a_lat else -param.m
    disorder = param.disorder*kwant.digest.gauss(site.tag, "")*1j
    return disorder*sigma_x + staggered_pot*sigma_0
def hopping(site1, site2, param):
    x1, y1 = site1.pos
    x2, y2 = site2.pos
    return -1*np.exp(-1j*param.phi/2*(x2 - x1)*(y1 + y2))*sigma_0
def nnn_hopping(site1, site2, param):
    x1, y1 = site1.pos
    x2, y2 = site2.pos
    return (sigma_0*np.real(param.t_2) + sigma_z*1j*np.imag(param.t_2))*np.exp(-1j*param.phi/2*(x2 - x1)*(y1 + y2))

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

def file_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape",          type=str,       default="triangular-zz",        choices=["ribbon", "rectangle", "triangular-zz", "triangular-ac", "hexagon-zz", "hexagon-ac", "circle"])
    parser.add_argument("--N_edge",         type=int,       default=25,                     help="Number of edges for triangular shapes")
    parser.add_argument("--r",              type=float,     default=11,                     help="Radius for circular and hexagonal shapes")
    parser.add_argument("--l",              type=float,     default=24,                     help="Length for triangular and hexagonal shapes")
    parser.add_argument("--Lx",             type=int,       default=15,                     help="Length in x direction for hexagonal shapes")
    parser.add_argument("--L",              type=float,     default=30.0,                   help="Length in y direction for rectangular shapes")
    parser.add_argument("--W",              type=float,     default=20.0,                   help="Width for rectangular shapes")
    parser.add_argument("--phi",            type=float,     default=0.0,                    help="Phase factor for hopping")
    parser.add_argument("--t_2",            type=float,     default=1e-12,                  help="NNN hopping parameter")
    parser.add_argument("--t_prime",        type=float,     default=0.0,                    help="NNN hopping imaginary parameter")
    parser.add_argument("--m",              type=float,     default=0.0,                    help="Staggered potential")
    parser.add_argument("--disorder",       type=float,     default=0.1,                    help="Disorder strength")
    parser.add_argument("--num_moments",    type=int,       default=50,                     help="Number of moments for KPM")
    parser.add_argument("--energy",         type=float,     default=0.0,                    help="Energy for KPM local spectral density plots")
    return parser.parse_args()

def main():
    shapes = ["triangular-zz", "ribbon", "triangular-ac", "hexagon-zz", "hexagon-ac", "circle", "rectangle"]
    
    args = file_input()

    parameters = SimpleNamespace(phi=args.phi, t_2=args.t_2+1j*args.t_prime, m=args.m, disorder=args.disorder)
    # shape = shapes[0]
    
    shape = args.shape    

    diminsions = SimpleNamespace(r = args.r, l = args.l, N_edge = args.N_edge, Lx = args.Lx, L=args.L, W=args.W)
    sys = make_system(1.0, shape, diminsions)
    final_sys = sys.finalized()

    h = final_sys.hamiltonian_submatrix(params=dict(param = parameters), sparse=False)
    N = h.shape[0]
    eig_vals, eig_states = np.linalg.eigh(h)
    # eig_vals, eig_states = sla.eigsh(h, N-2)
    eig_states= eig_states.T
    i0 = binary_search(eig_vals, 0.0)
    if np.allclose(eig_vals[i0], eig_vals[i0 - 1]):
        i0 -= 1
    
    # print("First unoccupied state:", i0)
    # plotand = np.abs(eig_states[i0][0::2])**2+np.abs(eig_states[i0][1::2])**2
    # kwant.plotter.map(final_sys, plotand, oversampling=10)
    # plt.show()
    # print("Highest occupied state:", i0+1)
    # plotand = np.abs(eig_states[i0][0::2])**2+np.abs(eig_states[i0][1::2])**2
    # kwant.plotter.map(final_sys, plotand, oversampling=10)
    # plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))



    # First unoccupied state
    plotand1 = np.abs(eig_states[i0][0::2])**2 + np.abs(eig_states[i0][1::2])**2 + np.abs(eig_states[i0+1][0::2])**2 + np.abs(eig_states[i0+1][1::2])**2
    kwant.plotter.map(final_sys, plotand1, ax=axes[0], oversampling=10, show=False)
    axes[0].set_title("First unoccupied state")

    # Highest occupied state
    plotand2 = np.abs(eig_states[i0 + 2][0::2])**2 + np.abs(eig_states[i0+2][1::2])**2 + np.abs(eig_states[i0 + 3][0::2])**2 + np.abs(eig_states[i0+3][1::2])**2
    kwant.plotter.map(final_sys, plotand2, ax=axes[1], oversampling=10, show=False)
    axes[1].set_title("Highest occupied state")

    plt.tight_layout()
    plt.show()

    # Calculate and plot spin-polarized currents using kwant.operator.Current for both states

    # Define the spin-z operator in orbital space
    current_op = kwant.operator.Current(final_sys, sigma_0)

    # Spin-z polarized current for the first unoccupied state
    spin_z_currents_1 = current_op(eig_states[i0], params=dict(param=parameters))
    kwant.plotter.current(final_sys, spin_z_currents_1, colorbar=True)

    # Spin-z polarized current for the highest occupied state
    spin_z_currents_2 = current_op(eig_states[i0 + 1], params=dict(param=parameters))
    kwant.plotter.current(final_sys, spin_z_currents_2, colorbar=True)

    # fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # # First unoccupied state current
    # kwant.plotter.current(final_sys, spin_z_currents_1, ax=axes2[0], colorbar=True, show=False)
    # axes2[0].set_title("First unoccupied state")

    # # Highest occupied state current
    # kwant.plotter.current(final_sys, spin_z_currents_2, ax=axes2[1], colorbar=True, show=False)
    # axes2[1].set_title("Highest occupied state")
    # # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()

