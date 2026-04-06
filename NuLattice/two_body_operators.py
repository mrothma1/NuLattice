import numpy as np
import itertools
import scipy.sparse as sparse
import sys
from concurrent.futures import ProcessPoolExecutor

import NuLattice.lattice as lat
import NuLattice.constants as consts
import NuLattice.one_body_operators as ob_ops

def f_SL(site1, site2, myL, sL):
    """
    Calculates the local smearing function at two points with strength sL
    
    :param site1:   first site on the lattice
    :type site1:    (int, int, int)
    :param site2:   second site on the lattice
    :type site2:    (int, int, int)
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param sL:      local smearing strength
    :type sL:       float
    :return:        value of the local smearing function
    :rtype:         float
    """
    if site1 == site2:
        return 1
    i1, j1, k1 = site1
    i2, j2, k2 = site2
    dist_sq = (((i1 - i2 + myL // 2) % myL - myL // 2) ** 2 
            + ((j1 - j2 + myL // 2) % myL - myL // 2) ** 2 
            + ((k1 - k2 + myL // 2) % myL - myL // 2) ** 2)
    if dist_sq == 1:
        return sL
    return 0

def rho_mult_NO(rho_1, rho_2, mult, max_mem = 0):
    """
    Multiplies the given density operators normal orders the product
    
    :param rho_1:   Array of first density operator
    :type rho_1:    scipy.sparse.coo_array
    :param rho_2:   Array of second density operator
    :type rho_2:    scipy.sparse.coo_array
    :param mult:    Factor to scale the calculation by
    :type mult:     float
    :param max_mem: Optional; Maximum memory size for the value array
                    to get to before compressing into a sparse format.
                    If left as 0, no limit to the memory will be set
    :type max_mem:  int
    :returns:       The density operators multilpied and normal ordered,
                    for V^{ij}_{kl}, it only stores i<j and k<l
    :rtype:         scipy.sparse.csr_array
    """
    tmp_col = []
    tmp_row = []
    tmp_val = []   
    dim = rho_1.shape[0]
    ret = sparse.csr_array((dim ** 2, dim ** 2))    
    for a, c, v in zip(rho_1.row, rho_1.col, rho_1.data):
        for b, d, w in zip(rho_2.row, rho_2.col, rho_2.data):                
            matele = mult * v * w
            if a < b and c < d:
                tmp_col.append(a + b * dim)
                tmp_row.append(c + d * dim)
                tmp_val.append(matele)
            if b < a and c < d:
                tmp_col.append(b + a * dim)
                tmp_row.append(c + d * dim)
                tmp_val.append(-matele)
                if max_mem != 0 and sys.getsizeof(tmp_val) > max_mem:
                    ret += sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim ** 2, dim ** 2))
                    tmp_val = []
                    tmp_row = []
                    tmp_col = []

    return ret + sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim ** 2, dim ** 2))

def shortRangeV_2body(lattice, myL, sL, sNL, c0, a_lat, op1b = None, spin = 2, isospin = 2, verbose = False, max_mem = 0):
    """
    Generates a short range two body interaction of the form sum_n :rho(n):^2 where rho is a smeared density
    
    :param lattice: list of lattice sites returned by lattice.get_lattice
    :type lattice:  list[(int, int, int)] 
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param sL:      local smearing strength
    :type sL:       float
    :param sNL:     non-local smearing strength
    :type sNL:      float
    :param c0:      strength of the interaction
    :type c0:       float
    :param a_lat:   lattice spacing divided by hbar c
    :type a_lat:    float
    :param op1b:    Optional; one body operator used to generate rho in the form 
                    a^dagger [op1b] a. If None, then the identity operator is used
    :type op1b:     scipy.sparse.csr_array()
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int  
    :param verbose: Optional; whether or not to print progress during calculation
    :type verbose:  bool
    :param max_mem: maximum memory size for the temporary float array to get to before 
                    compressing it into a scipy.sparse array
    :type max_mem:  float
    :return:        A sparse csr_array representing V^{ab}_{ij} in MeV, 
                    for dim=spin*isospin*L^3, the row indicies correspond
                    to a + b * dim + c and column indices correspond to i + j * dim
    :rtype:         scipy.sparse.csr_array()
    """
    rho_n = []
    if verbose:
        print('Generating Densities...',end='')
    for site in lattice:
        rho_n.append(ob_ops.rho_op(site, myL, op1b=op1b, sNL=sNL, spin=spin, isospin = isospin))
    if verbose:
        print('Done')

    dim  = myL **3 * spin * isospin
    if sL != 0:
        if verbose:
            print('Performing Local Smearing...',end='')
        rho_smeared = []
        for site1 in lattice:
            tmp = sparse.csr_array(np.zeros([dim, dim]))
            for site2 in lattice:
                pos = site2[0] * myL ** 2 + site2[1] * myL + site2[2]
                scale = f_SL(site1, site2, myL, sL)
                if scale != 0:
                    tmp += rho_n[pos] * scale
            rho_smeared.append(tmp.tocoo())
        if verbose:
            print('Done')
    else:
        rho_smeared = rho_n
    if verbose:
        print('Generating Interaction...',end='')
    ret = sparse.csr_array((dim ** 2, dim ** 2))
    with ProcessPoolExecutor() as executor:
        size = len(rho_smeared)
        for val in executor.map(rho_mult_NO, rho_smeared, rho_smeared, [c0 / a_lat] * size, [max_mem] * size):
            ret += val
    if verbose:
        print('Interaction Generated')

    return ret

def f_SS(myL, bpi, a_lat):
    """
    f_S'S function to be used in one pion exchange
    
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param bpi:     Parameter to remove the short-distance lattice artifacts
    :type bpi:      float
    :param a_lat:   lattice spacing divided by hbar c
    :type a_lat:    float
    :return:        f_S'S(m), where m is the vector n'-n, as a numpy array that gets 
                    indexed as fSS[S'][S][m_x][m_y][m_z]
    :rtype:         complex numpy array of dimension 3,3, myL, myL, myL
    """
    m_pi = consts.m_pi_0 * a_lat
    q =np.fft.fftfreq(myL, 1/ (2.0 * np.pi))
    qx, qy, qz = np.meshgrid(q, q, q, indexing="ij")

    q2 = qx**2 + qy**2 + qz**2
    ft_fss = np.exp(-bpi * q2) / (q2 + m_pi**2)

    q = np.array([qx, qy, qz])

    fSS = np.zeros((3, 3, myL, myL, myL), dtype=complex)

    for s1, s2 in itertools.product(range(3), range(3)):
        fSS[s1, s2] = np.fft.ifftn(q[s1] * q[s2] * ft_fss)

    return fSS

def onePionEx(myL, bpi, a_lat, lattice, verbose = False, mult = 1,max_mem=1e8):
    """
    computes the potential for one pion exchange

    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param bpi:     parameter to remove short-distance lattice artifacts
    :type bpi:      float
    :param a_lat:   lattice spacing divided by hbar c
    :type a_lat:    float
    :param lattice: list of lattice sites returned by lattice.get_lattice
    :type lattice:  list[(int, int, int)] 
    :param verbose: Optional; whether or not to print progress during calculation
    :type verbose:  bool
    :param mult:    Optional; scale factor to multiply potential by
    :type mult:     float
    :param max_mem: maximum memory size for the temporary float array to get to 
                    before compressing it into a scipy.sparse array
    :type max_mem:  float
    :return:        A sparse csr_array representing V^{ab}_{ij} in MeV with the 
                    row indicies corresponding to a + b * 4*L^3 and 
                    column indices corresponding to i + j * 4*L^3
    :rtype:         scipy.sparse.csr_array()
    """
    if verbose:
        print('Calculating f_SS...', end='')
    fSS = f_SS(myL, bpi, a_lat)
    scale = - (consts.g_A / (2.0 * a_lat * consts.f_pi)) ** 2 * mult / 2.0
    dim  = myL **3 * 4
    if verbose:
        print('Done\nCalculating Densities...', end='')
    dens = [None] * myL ** 3
    sp_ops = [ob_ops.list_to_sparse1b(ob_ops.spin_x(lattice, myL)), 
            ob_ops.list_to_sparse1b(ob_ops.spin_y(lattice, myL)), 
            ob_ops.list_to_sparse1b(ob_ops.spin_z(lattice, myL))]
    iso_ops = [ob_ops.list_to_sparse1b(ob_ops.tau_x(lattice, myL)), 
           ob_ops.list_to_sparse1b(ob_ops.tau_y(lattice, myL)), 
           ob_ops.list_to_sparse1b(ob_ops.tau_z(lattice, myL))]
    for site in lattice:
        rho_sp = [None] * 3
        for sp in range(3):
            rho_sp_iso = [None] * 3
            for iso in range(3):
                op1b = sp_ops[sp] @ iso_ops[iso]
                rho_sp_iso[iso] =ob_ops.rho_op(site, myL, op1b,op_fac = 4.0, spin=2, isospin=2)
            rho_sp[sp] = rho_sp_iso
        loc = lat.site2index(site, myL)
        dens[loc] = rho_sp
    sparse_full_ope = sparse.csr_array((dim * dim, dim * dim))
    tmp_val = []
    tmp_row = []
    tmp_col = []
    if verbose:
        print('Done\nGenerating Interaction...')

    for sp1, sp2, iso, site1, site2 in itertools.product(range(3), range(3), 
                                                         range(3), lattice, lattice):
        f_SS_val = fSS[sp1, sp2, (site1[0] - site2[0]) % myL, (site1[1] - site2[1]) % myL, 
                  (site1[2] - site2[2]) % myL]
        if f_SS_val == 0:
            continue
        loc1 = lat.site2index(site1, myL)
        loc2 = lat.site2index(site2, myL)
        rho1 = dens[loc1][sp1][iso]
        rho2 = dens[loc2][sp2][iso]
        
        for a, c, v in zip(rho1.row, rho1.col, rho1.data):
            for b, d, w in zip(rho2.row, rho2.col, rho2.data):     
                matele = f_SS_val * v * w * scale / a_lat
                if a < b and c < d:
                    tmp_col.append(a + b * dim)
                    tmp_row.append(c + d * dim)
                    tmp_val.append(matele)
                if b < a and c < d:
                    tmp_col.append(b + a * dim)
                    tmp_row.append(c + d * dim)
                    tmp_val.append(-matele)
                if a > b and c > d:
                    tmp_col.append(b + a * dim)
                    tmp_row.append(d + c * dim)
                    tmp_val.append(matele)
                if b > a and c > d:
                    tmp_col.append(a + b * dim)
                    tmp_row.append(d + c * dim)
                    tmp_val.append(-matele)
                if max_mem != 0 and sys.getsizeof(tmp_val) > max_mem:
                    if verbose:
                        print(f'Compressing interaction...',end='')
                    sparse_full_ope += sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim * dim, dim * dim))
                    tmp_val = []
                    tmp_row = []
                    tmp_col = []
                    if verbose:
                        print('Compressed')
    if len(tmp_col) > 0:
        if verbose:
            print(f'Compressing interaction...',end='')
        sparse_full_ope += sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim * dim, dim * dim))
        if verbose:
            print('Done')
    if verbose:
        print('Interaction Generated')

    return sparse_full_ope

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
                    the row indicies correspond to a + b * dim + c and column 
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
    rho_over_d = []
    for site1 in lattice:
        tmp = sparse.csr_array(np.zeros([dim, dim]))
        for site2 in lattice:
            pos = site2[0] * myL ** 2 + site2[1] * myL + site2[2]
            if site1 == site2:
                scale = 2.0
            else:
                dist_sq = ((site1[0]- site2[0] + myL // 2) % myL - myL // 2) ** 2 + ((site1[1]- site2[1] + myL // 2) % myL - myL // 2) ** 2 + ((site1[2]- site2[2] + myL // 2) % myL - myL // 2) ** 2
                scale = 1 / np.sqrt(dist_sq)
            if scale != 0:
                tmp += rho_n[pos] * scale
        rho_over_d.append(tmp.tocoo())
    if verbose:
        print('Done\nGenerating Interaction...',end='')
    ret = sparse.csr_array((dim ** 2, dim ** 2))
    with ProcessPoolExecutor() as executor:
        size = len(rho_n)
        for val in executor.map(rho_mult_NO, rho_over_d, rho_n, [consts.alpha_EM / a_lat] * size, [max_mem] * size):
            ret += val
    if verbose:
        print('Interaction Generated')

    return ret

def sparse_to_list_2body(sparse_int, myL, spin=2, isospin=2):
    """
    Takes sparse two-body interaction and converts it to a list

    :param sparse_int:  Interaction stored as a sparse matrix
    :type sparse_int:   scipy.sparse.csr_array
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spin:         int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int  
    :return:            list of lists [a,b,c,d,v] where a, b, c, and d are 
                        indices and v is the value for V^ab_cd
    :rtype:             list[[int, int, int, int, float]]
    """
    ret = []
    dim  = myL ** 3 * spin * isospin
    sparse_int_coo = sparse_int.tocoo()
    for i, j, v in zip(sparse_int_coo.row, sparse_int_coo.col, sparse_int_coo.data):
        a = i % dim
        b = (i - a) // dim
        c = j % dim
        d = (j - c) // dim

        ret.append([a, b, c, d, v])
    return ret