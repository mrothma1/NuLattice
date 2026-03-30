import scipy.sparse as sparse
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import NuLattice.one_body_operators as ob_ops
import NuLattice.two_body_operators as tb_ops

def rho_cubed_NO(rho_n, mult, min_val, max_mem=0):
    """
    Computes the density cubed and normal ordered at a lattice site n
    
    :param rho_n:   Array storing the density at lattice site n
    :type rho_n:    scipy.sparse.coo_array
    :param mult:    Factor to scale the calculation by
    :type mult:     
    :param min_val: minimum value of rho_n to use in calculation, 
                    if rho(n) < min_val, it will be discarded
    :type min_val:  float
    :param max_mem: Optional; Maximum memory size for the value array
                    to get to before compressing into a sparse format.
                    If left as 0, no limit to the memory will be set
    :type max_mem:  int
    :returns:       The density operator squared and normal ordered, for
                    V^{ij}_{kl}, it only stores i<j and k<l
    :rtype:         scipy.sparse.csr_array
    """
    tmp_col = []
    tmp_row = []
    tmp_val = [] 
    dim = rho_n.shape[0]
    ret = sparse.csr_array((dim ** 3, dim ** 3))
    for a, d, v in zip(rho_n.row, rho_n.col, rho_n.data):
        if abs(v) < min_val:
            continue
        for b, e, w in zip(rho_n.row, rho_n.col, rho_n.data):
            if abs(w) < min_val:
                continue     
            for c, f, x in zip(rho_n.row, rho_n.col, rho_n.data):
                if abs(x) < min_val:
                    continue
                matele = mult * v * w * x
                if a < b and b < c and d < e and e < f:
                    tmp_col.append(a + b * dim + c * dim ** 2)
                    tmp_row.append(d + e * dim + f * dim ** 2)
                    tmp_val.append(matele)
                
                if b < a and a < c and d < e and e < f:
                    tmp_col.append(b + a * dim + c * dim ** 2)
                    tmp_row.append(d + e * dim + f * dim ** 2)
                    tmp_val.append(-matele)
                
                if c < b and b < a and d < e and e < f:
                    tmp_col.append(c + b * dim + a * dim ** 2)
                    tmp_row.append(d + e * dim + f * dim ** 2)
                    tmp_val.append(matele)
                
                if a < c and c < b and d < e and e < f:
                    tmp_col.append(a + c * dim + b * dim ** 2)
                    tmp_row.append(d + e * dim + f * dim ** 2)
                    tmp_val.append(-matele)

                if c < b and b < a and d < e and e < f:
                    tmp_col.append(c + b * dim + a * dim ** 2)
                    tmp_row.append(d + e * dim + f * dim ** 2)
                    tmp_val.append(-matele)

                if b < c and c < a and d < e and e < f:
                    tmp_col.append(b + c * dim + a * dim ** 2)
                    tmp_row.append(d + e * dim + f * dim ** 2)
                    tmp_val.append(matele)
                if max_mem != 0 and sys.getsizeof(tmp_val) > max_mem:
                    ret += sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim ** 3, dim ** 3))
                    tmp_val = []
                    tmp_row = []
                    tmp_col = []
    return ret + sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim ** 3, dim ** 3))

def shortRangeV_3body(lattice, myL, sL, sNL, c0, a_lat, spin = 2, isospin = 2, verbose = False, min_val = 0, max_threads = None, max_mem=0):
    """
    Generates the leading-order short-range interaction defined in equation 13
    
    :param lattice: list of lattice sites returned by lattice.get_lattice
    :type lattice:  list[(int, int, int)] 
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param sL:      local smearing strength
    :type sL:       float
    :param sNL:     non-local smearing strength
    :type sNL:      float
    :param c0:      strength of the short range interaction
    :type c0:       float
    :param a_lat: lattice spacing divided by hbar c
    :type a_lat: float
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int  
    :param verbose: Optional; whether or not to print progress during calculation
    :type verbose:  bool
    :param max_mem: maximum memory size for the temporary float array to get to before compressing it into a scipy.sparse array
    :type max_mem:  float
    :return:        A sparse csr_array representing V^{ab}_{ij} in MeV with the row 
                    indicies corresponding to a + b * spin*isospin*L^3 and column 
                    indices corresponding to i + j * spin*isospin*L^3
    :rtype:         scipy.sparse.csr_array()
    """
    #list of density matrices
    rhos = []
    if verbose:
        print('Generating Densities...',end='')
    for site in lattice:
        rhos.append(ob_ops.rho_op(site, myL, sNL=sNL, spin=spin, isospin = isospin))
    if verbose:
        print('Done\nGenerating rho * f_SL...',end='')

    dim  = myL **3 * spin * isospin
    #List of sum_n rho(n)f_SL(m)
    rho_fsl = []
    for site1 in lattice:
        tmp = sparse.csr_array(np.zeros([dim, dim]))
        for site2 in lattice:
            pos = site2[0] * myL ** 2 + site2[1] * myL + site2[2]
            scale = tb_ops.f_SL(site1, site2, myL, sL)
            if scale != 0:
                tmp += rhos[pos] * scale
        rho_fsl.append(tmp.tocoo())
    if verbose:
        print('Done\nGenerating Interaction...',end='')

    ret = sparse.csr_array((dim ** 3, dim ** 3))
    with ProcessPoolExecutor(max_workers=max_threads) as executor:
        size = len(rho_fsl)
        for val in executor.map(rho_cubed_NO, rho_fsl, [c0 / a_lat] * size, [min_val] * size, [max_mem] * size):
            ret += val
    if verbose:
        print('Interaction Generated')
    return ret

def sparse_to_list_3body(sparse_int, myL, spin=2, isospin=2):
    """
    Takes sparse 3-body interaction and converts it to a list

    :param sparse_int:  Interaction stored as a sparse matrix
    :type sparse_int:   scipy.sparse.csr_array
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spin:         int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int  
    :return:            list of lists [a,b,c,d,e,f,v] where a, b, c, d, e, and f
                        are indices and v is the value for V^abc_def
    :rtype:             list[[int, int, int, int, int, int float]]
    """
    ret = []
    dim  = myL ** 3 * spin * isospin
    sparse_int_coo = sparse_int.tocoo()
    for i, j, v in zip(sparse_int_coo.row, sparse_int_coo.col, sparse_int_coo.data):
        a = i % dim
        b = ((i - a) // dim) % dim
        c = (((i - a) // dim) - b) // dim
        d = j % dim
        e = ((j - d) // dim) % dim
        f = (((j - d) // dim) - e) // dim
        ret.append([a, b, c, d, e, f, v])