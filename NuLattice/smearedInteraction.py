"""
Module that provides functions that generate a smeared interaction
"""

import numpy as np
from opt_einsum import contract
import copy
import NuLattice.lattice as lat
import NuLattice.constants as consts
import math
import itertools
import NuLattice.FCI.few_body_diagonalization as fbd
import scipy.sparse as sparse
import sys

def f_SL(site1, site2, myL, sL):
    """
    Calculates the local smearing function defined by equation 14
    
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
    dist = ((i1 - i2 + myL // 2) % myL - myL // 2) ** 2 + ((j1 - j2 + myL // 2) % myL - myL // 2) ** 2 + ((k1 - k2 + myL // 2) % myL - myL // 2) ** 2
    if dist== 1:
        return sL
    return 0
    
def indConv(ind, myL):
    """
    Gets x,y,z indices from combined index
    
    :param ind: index in lattice, given by x + y * myL + z * myL ^ 2
    :type ind:  int
    :param myL: number of lattice sites in each direction
    :type myL:  int
    :return:    list of x, y, z
    :rtype:     list[int]
    """
    x = ind % myL
    y = ((ind - x) // myL) % myL
    z = ((ind - x) // myL - y) // myL
    return x, y, z

def nonLocOp(site, myL, sNL, sz, tz,spin=2,isospin=2):
    """
    Generates the non-local creation/annihilation operator as definedin equations 3/4
    
    :param site:    x,y,z coordinate on the lattice
    :type site:     [int,int,int]
    :param myL:     size of lattice
    :type myL:      int
    :param sNL:     strength of the non-local smearing
    :type sNL:      float
    :param sz:      spin
    :type sz:       int
    :param tz:      isospin
    :type sz:       int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :return:        1D list equivalent to the non-local creation/annihilation operator at
                    a given site
    :rtype:         numpy list[float]
    """
    ret = np.zeros(myL ** 3*spin*isospin)
    pos = (site[0] * myL ** 2 + site[1] * myL + site[2])* spin * isospin + tz * spin + sz
    rx = (lat.right(site[0], myL) * myL ** 2 + site[1] * myL + site[2])* spin * isospin + tz * spin + sz
    ry = (lat.right(site[1], myL) * myL + site[0] * myL ** 2 + site[2])* spin * isospin + tz * spin + sz
    rz = (lat.right(site[2], myL) + site[0] * myL ** 2 + site[1] * myL)* spin * isospin + tz * spin + sz
    lx = (lat.left(site[0], myL) * myL ** 2 + site[1] * myL + site[2])* spin * isospin + tz * spin + sz
    ly = (lat.left(site[1], myL)  * myL + site[0] * myL ** 2 + site[2])* spin * isospin + tz * spin + sz
    lz = (lat.left(site[2], myL) + site[0] * myL ** 2 + site[1] * myL)* spin * isospin + tz * spin + sz
    ret[pos] += 1
    ret[rx] += sNL
    ret[lx] += sNL
    ret[ry] += sNL
    ret[ly] += sNL
    ret[rz] += sNL
    ret[lz] += sNL
    return ret

def densNonLoc(site, myL, sNL, spin=2, isospin=2):
    """
    Generates the non-local density operator as defined in equation 9
    
    :param site:    location in lattice
    :type site:     [int, int, int]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param sNL:     non-local smearing strength
    :type sNL:      float
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int  
    :returns:       non-local density operator for the given site
    :rtype:         scipy.sparse.csr_array()  
    """
    ret = sparse.csr_array(np.zeros([myL ** 3 * spin * isospin, myL ** 3 * spin * isospin]))
    for sz in range(spin):
        for tz in range(isospin):
            obo = nonLocOp(site, myL, sNL, sz, tz, spin, isospin)
            ret += sparse.csr_array(np.outer(obo, obo))
    return ret

def shortRangeV(lattice, myL, sL, sNL, c0, spin = 2, isospin = 2):
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
    :param c0: Description
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int  
    :return:        list of lists [i, j, k, l, value] where i, j and k, l are indices of two particles
                    in the single-particle basis, and value is the value of the matrix element <ij||kl>. 
                    All elements have i<j and k<l
    :rtype:         list[(int, int, int, int, float)]
    """
    ret = []


    #list of density matrices
    rhos = []
    
    for site in lattice:
        rhos.append(densNonLoc(site, myL, sNL, spin, isospin))
    print('Densities Generated')

    dim  = myL **3 * spin * isospin
    #List of sum_n rho(n)f_SL(m)
    rho_fsl = []
    for site1 in lattice:
        tmp = sparse.csr_array(np.zeros([dim, dim]))
        for site2 in lattice:
            pos = site2[0] * myL ** 2 + site2[1] * myL + site2[2]
            scale = f_SL(site1, site2, myL, sL)
            if scale != 0:
                tmp += rhos[pos] * scale
        rho_fsl.append(tmp.tocoo())
    print('rho*f_SL Generated')
    
    sparse_full_int = sparse.csr_array((dim * dim, dim * dim))
    #Fully compute eq 13
    tmp_col = []
    tmp_row = []
    tmp_val = []
    for optempn in rho_fsl:        
        for a, c, v in zip(optempn.row, optempn.col, optempn.data):
            for b, d, w in zip(optempn.row, optempn.col, optempn.data):                
                matele = c0 * v * w
                if a < b and c < d:
                    tmp_col.append(a + b * dim)
                    tmp_row.append(c + d * dim)
                    tmp_val.append(matele)
                if b < a and c < d:
                    tmp_col.append(b + a * dim)
                    tmp_row.append(c + d * dim)
                    tmp_val.append(-matele)
                if sys.getsizeof(tmp_val) >= 1e8:
                    print(f'Compressing interaction...',end=' ')
                    sparse_full_int += sparse.coo_array((tmp_val, (tmp_row, tmp_col)), shape = (dim * dim, dim * dim))
                    sparse_full_int.sum_duplicates()
                    tmp_val = []
                    tmp_row = []
                    tmp_col = []
                    print('Compressed')
    if len(tmp_col) > 0:
        print(f'Compressing interaction...',end=' ')
        sparse_full_int += sparse.coo_array((tmp_val, (tmp_row, tmp_col)), shape = (dim * dim, dim * dim))
        sparse_full_int.sum_duplicates()
        print('Compressed')
    print('Converting to List')
    del(tmp_val)
    del(tmp_row)
    del(tmp_col)

    sparse_full_int = sparse_full_int.tocoo()
    for i, j, v in zip(sparse_full_int.row, sparse_full_int.col, sparse_full_int.data):
        a = i % dim
        b = (i - a) // dim
        c = j % dim
        d = (j - c) // dim

        ret.append([a, b, c, d, v])
            
    print('Full Interaction Generated')

    return ret

def get_pauli(ind):
    """
    Gets the one of the 3 Pauli matrices based on the index given
    
    :param ind: Index of which of the 3 Pauli matrices to get (1, 2, or 3)
    :type ind:  int
    :return:    One of the 3 Pauli matrices
    :rtype:     2x2 numpy array(complex)
    """
    ret = np.zeros([2, 2], dtype=complex)
    if ind == 1:
        ret[0,1] = 1
        ret[1,0] = 1
    elif ind == 2:
        ret[0,1] = -1j
        ret[1,0] = 1j
    elif ind == 3:
        ret[0,0] = 1
        ret[1,1] = -1
    else:
        print('Index must be between 1 and 3')
        return -1
    return ret

def dens_spi_iso(site, myL, spI, isoI):
    """
    Generates the spin, isospin density operator at a lattice site as defined in equation 8
    
    :param site:    location in lattice
    :type site:     [int, int, int]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param spI:     spin index
    :type spI:      int
    :param isoI:    isospin index
    :type isoI:     int
    :return:        density operator as a sparse matrix
    :rtype:         scipy.sparse.csr_array(complex)
    """
    spin = 2
    isospin = 2
    ret = sparse.csr_array(np.zeros([myL ** 3 * 2 * isospin, myL ** 3 * spin * isospin]))

    pauli_s = get_pauli(spI)
    pauli_i = get_pauli(isoI)
    tensProd = np.tensordot(pauli_s, pauli_i, axes=0)
    loc = (site[0] * myL ** 2 + site[1] * myL + site[2])* spin * isospin
    for ind, ele in np.ndenumerate(tensProd):
        if ele == 0:
            continue
        sz1, sz2, tz1, tz2 = ind
        obo1 = np.zeros(myL ** 3*spin*isospin)
        pos = loc + tz1 * spin + sz1
        obo1[pos] = 1
        obo2 = np.zeros(myL ** 3*spin*isospin)
        pos = loc + tz2 * spin + sz2
        obo2[pos] = 1
        ret += sparse.csr_array(ele * np.outer(obo1, obo2))
    return ret

def ope(myL, bpi, a_lat):
    return 0
def onePionEx(myL, bpi, spin, a_lat):
    """
    computes the potential for one pion exchange

    :param myL: number of lattice sites in each direction
    :type myL:  int
    :param bpi: parameter to remove short-distance lattice artifacts
    :type bpi: float
    :param spin: Not sure why this is specified tbh, only spin 0 is used in the program
    :type spin: int
    :param a_lat: lattice spacing divided by hbar c
    :type a_lat: float
    """
    coupling = (-(consts.g_A / (2 * a_lat * consts.f_pi)) ** 2)
    mass = consts.m_pi * a_lat
    cg = np.zeros([2, 2, 2, 4])
    val = 1 / np.sqrt(2)
    cg[0, 1, 0, 0] = val
    cg[1, 0, 0, 0] = -val
    cg[0,0,1,2] = 1.0
    cg[0,1,1,1] = val
    cg[1,0,1,1] = val
    cg[1,1,1,0] = 1.0 

    pauli = np.zeros([2, 2, 3], dtype=complex)
    pauli[0, 1, 0] = 1.0
    pauli[1, 0, 0] = 1.0
    pauli[0, 1, 1] = -1j
    pauli[1, 0, 1] = 1j
    pauli[0, 0, 2] = 1.0
    pauli[1, 1, 2] = -1.0

    pp = (np.linspace(0,myL**3-1, num=myL**3)).astype(int)
    px = pp % myL
    py = ((pp - px) / myL) % myL
    pz = (pp-px-py*myL) / myL ** 2
    px = ((px+myL/2.0) % myL) - myL/2.0
    py = ((py+myL/2.0) % myL) - myL/2.0
    pz = ((pz+myL/2.0) % myL) - myL/2.0
    
    val = 2 * np.pi / myL
    q2 = ((np.square(px) + np.square(py) + np.square(pz)) * (val) ** 2).astype(float)
    qqx = val * px
    qqy = val * py
    qqz = val * pz

    potOPE = [[0 for _ in range(2 * spin + 1)] for _ in range(2 * spin + 1)]
    for sz in range(-spin, spin+1):
        for szp in range(-spin, spin+1):
            potMom = (np.zeros([1, myL**3])).astype(complex)
            for nSpin in range(16):
                is1 = int(nSpin % 2)
                is2 = int(((nSpin - is1) % 4) / 2)
                is1p = int(((nSpin - is1 - 2 * (is2)) % 8) / 4)
                is2p = int((nSpin - is1 - 2 * (is2) - 4 * (is1p)) / 8)
                potMom +=   (coupling 
                            *cg[is1, is2, spin, sz + spin] 
                            *cg[is1p, is2p, spin, szp + spin]
                            *(pauli[is1, is1p, 0] * qqx 
                                     + pauli[is1, is1p, 1] * qqy 
                                     + pauli[is1, is1p, 2] * qqz)
                            *(pauli[is2, is2p, 0] * qqx 
                                     + pauli[is2, is2p, 1] * qqy 
                                     + pauli[is2, is2p, 2] * qqz)
                            /(q2 + mass ** 2) * np.exp(-bpi * (q2 + mass ** 2))
                            )
            potMom_reshaped = np.reshape(potMom, (myL, myL, myL))
            ftPotMom = np.fft.ifftn(potMom_reshaped)
            ftPotMom_reshaped = np.reshape(ftPotMom, (myL**3))
            potOPE[sz + spin][szp + spin] = ftPotMom_reshaped
    
    return (np.diag((potOPE[0][0])))

def tKin(myL, Nk, a, spin=2, isospin=2):
    """
    computes 1-body kinetic energy matrix elements.
    
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param Nk:      number of neighbors along each axis to use(1 for nearest-neighbor, 2 for
                    next-to-nearest-neighbor, etc)
    :type Nk:       int
    :param a:       lattice spacing in dimensionless lattice units
    :type a:        float
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int    
    :return:        list of tuples [i, j, value] where i and j are indices in the single-particle 
                    basis, and value is the value of the matrix element Tij
    :rtype:         list[(int, int, float)]
    """
    h = -1.0 / 2.0 / (consts.mass*a)

    KK = np.zeros([myL**3, myL**3])
    cf0 = 0.0

    r = np.arange(myL**3)
    nx = np.mod(r, myL)
    ny = np.mod((r - nx) // myL, myL)
    nz = (r - nx - ny * myL) // (myL**2)

    for k in range(1, Nk + 1):
        cf = ((-1)**(k + 1) * 2.0 *
              (math.factorial(Nk) / math.factorial(Nk - k)) /
              (math.factorial(Nk + k) /math.factorial(Nk)) / k**2 * h)
        cf0 -= 2 * cf

        rxp = np.mod(nx + k, myL) + ny * myL + nz * myL**2
        rxm = np.mod(nx - k, myL) + ny * myL + nz * myL**2
        ryp = nx + np.mod(ny + k, myL) * myL + nz * myL**2
        rym = nx + np.mod(ny - k, myL) * myL + nz * myL**2
        rzp = nx + ny * myL + np.mod(nz + k, myL) * myL**2
        rzm = nx + ny * myL + np.mod(nz - k, myL) * myL**2

        KK[r, rxp] += cf
        KK[r, rxm] += cf
        KK[r, ryp] += cf
        KK[r, rym] += cf
        KK[r, rzp] += cf
        KK[r, rzm] += cf

    cf0 *= 3
    KK[r, r] += cf0
    ret = tKinMatToList(KK, myL, spin, isospin)
    return ret

def tKinMatToList(tMat, myL, spin = 2, isospin =2):
    """
    Takes kinetic energy from a matrix to a list of indices and values
    
    :param tMat:    kinetic energy matrix
    :type tMat:     list[list[float]]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :return:        list of tuples [i, j, value] where i and j are indices in the single-particle 
                    basis, and value is the value of the matrix element Tij
    :rtype:         list[(int, int, float)]
    """
    ret = []
    for i in range(myL ** 3):
        for j in range(myL ** 3):
            val = tMat[i][j]
            if val == 0:
                continue
            indx1, indy1, indz1 = indConv(i, myL)
            indx2, indy2, indz2 = indConv(j, myL)
            for tz in range(isospin):
                for sz in range(spin):
                    state1 = [indx1, indy1, indz1, tz, sz]
                    ind1 = lat.state2index(state1, myL, spin, isospin)
                    state2 = [indx2, indy2, indz2, tz, sz]
                    ind2 = lat.state2index(state2, myL, spin, isospin)
                    ret.append([ind1, ind2, val])
    return ret
