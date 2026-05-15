import numpy as np
import itertools
import scipy.sparse as sparse
import sys
from concurrent.futures import ProcessPoolExecutor

import NuLattice.lattice as lat
import NuLattice.constants as consts
import NuLattice.operators.one_body_operators as ob_ops
import NuLattice.operators.two_body_operators as twob_ops

def coulomb_pot(lattice, myL, a_lat, spin = 2, isospin = 2, verbose = False, max_mem=0):
    """
    Creates the coulomb potential

    :param lattice: list of lattice sites returned by lattice.get_lattice
    :type lattice:  list[(int, int, int)] 
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param a_lat:   lattice spacing divided by hbar c
    :type a_lat:    float
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int  
    :param verbose: Optional; whether or not to print progress during calculation
    :type verbose:  bool
    :param max_mem: maximum memory size for the temporary float array to get to before compressing it into a scipy.sparse array
    :type max_mem:  float
    :return:        A sparse csr_array representing V^{ab}_{ij} in MeV, for dim=spin*isospin*L^3,
                    the row indicies correspond to a + b * dim and column 
                    indices correspond to i + j * dim
    :rtype:         scipy.sparse.csr_array()
    """
    rho_n = []
    if verbose:
        print('Generating Densities...',end='')
    tau_3 = 2.0 * ob_ops.list_to_sparse1b(ob_ops.tau_z(lattice, myL))
    for site in lattice:
        rho_n.append((ob_ops.rho_op(site, myL, sNL=0, spin=spin, isospin = isospin)-ob_ops.rho_op(site, myL, sNL=0, op1b = tau_3,spin=spin, isospin = isospin)).tocoo())
    if verbose:
        print('Done\nGenerating rho(n) / d(n-n\')...',end='')

    dim  = myL **3 * spin * isospin
    inv_dist = np.zeros([dim, dim])
    for site1 in lattice:
        for site2 in lattice:
            pos1 = site1[0] * myL ** 2 + site1[1] * myL + site1[2]
            pos2 = site2[0] * myL ** 2 + site2[1] * myL + site2[2]
            if site1 == site2:
                val = 2.0
            else:
                dist_sq = ((site1[0]- site2[0] + myL // 2) % myL - myL // 2) ** 2 + ((site1[1]- site2[1] + myL // 2) % myL - myL // 2) ** 2 + ((site1[2]- site2[2] + myL // 2) % myL - myL // 2) ** 2
                val = 1 / np.sqrt(dist_sq)
            inv_dist[pos1, pos2] = val
    if verbose:
        print('Done\nGenerating Interaction...',end='')
    ret = sparse.csr_array((dim ** 2, dim ** 2))
    with ProcessPoolExecutor() as executor:
        size = len(rho_n)
        for site in lattice:
            pos = site[0] * myL ** 2 + site[1] * myL + site[2]
            rho_n_prime = rho_n[pos]
            scale = consts.alpha_EM / a_lat * inv_dist[pos,:] / 4.0
            for val in executor.map(twob_ops.rho_mult_NO, [rho_n_prime] * size, rho_n, scale, [max_mem] * size):
                ret += val
    if verbose:
        print('Interaction Generated')

    return ret