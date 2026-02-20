"""
Module that provides functions that generate a smeared interaction
"""

import numpy as np
import NuLattice.lattice as lat
import NuLattice.constants as consts
import math
import itertools
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
    pos = lat.state2index([site[0], site[1], site[2], tz, sz], myL, spin, isospin)
    rx = lat.state2index([lat.right(site[0], myL), site[1], site[2], tz, sz], myL, spin, isospin)
    ry = lat.state2index([site[0], lat.right(site[1], myL), site[2], tz, sz], myL, spin, isospin)
    rz = lat.state2index([site[0], site[1], lat.right(site[2], myL), tz, sz], myL, spin, isospin)
    lx = lat.state2index([lat.left(site[0], myL), site[1], site[2], tz, sz], myL, spin, isospin)
    ly = lat.state2index([site[0], lat.left(site[1], myL), site[2], tz, sz], myL, spin, isospin)
    lz = lat.state2index([site[0], site[1], lat.left(site[2], myL), tz, sz], myL, spin, isospin)
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

def shortRangeV(lattice, myL, sL, sNL, c0, a_lat, spin = 2, isospin = 2, verbose = False):
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
    :return:        list of lists [i, j, k, l, value] where i, j and k, l are indices of two particles
                    in the single-particle basis, and value is the value of the matrix element <ij||kl> in MeV. 
                    All elements have i<j and k<l
    :rtype:         list[(int, int, int, int, float)]
    """
    ret = []


    #list of density matrices
    rhos = []
    if verbose:
        print('Generating Densities...',end='')
    for site in lattice:
        rhos.append(densNonLoc(site, myL, sNL, spin, isospin))
    if verbose:
        print('Done\nGenerating rho * f_SL...',end='')

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
    if verbose:
        print('Done\nGenerating Interaction...')
    
    sparse_full_int = sparse.csr_array((dim * dim, dim * dim))
    #Fully compute eq 13
    tmp_col = []
    tmp_row = []
    tmp_val = []
    for optempn in rho_fsl:        
        for a, c, v in zip(optempn.row, optempn.col, optempn.data):
            for b, d, w in zip(optempn.row, optempn.col, optempn.data):                
                matele = c0 * v * w / a_lat
                if a < b and c < d:
                    tmp_col.append(a + b * dim)
                    tmp_row.append(c + d * dim)
                    tmp_val.append(matele)
                if b < a and c < d:
                    tmp_col.append(b + a * dim)
                    tmp_row.append(c + d * dim)
                    tmp_val.append(-matele)
                if sys.getsizeof(tmp_val) >= 1e8:
                    if verbose:
                        print(f'Compressing interaction...',end='')
                    sparse_full_int += sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim * dim, dim * dim))
                    sparse_full_int.sum_duplicates()
                    tmp_val = []
                    tmp_row = []
                    tmp_col = []
                    if verbose:
                        print('Done')
    if len(tmp_col) > 0:
        if verbose:
            print(f'Compressing interaction...',end='')
        sparse_full_int += sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim * dim, dim * dim))
        sparse_full_int.sum_duplicates()
        if verbose:
            print('Done')
    if verbose:
        print('Interaction Generated\nConverting to List...',end='')
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
    if verbose:
        print('Done')

    return ret

def get_spin_op(ind, dof):
    """
    Gets the one of the 3 spin matrices based on the index given
    
    :param ind: Index of which of the 3 spin matrices to get (0, 1, or 2 for x, y, and z respectively)
    :type ind:  int
    :param dof: total spin/isospin degrees of freedon
    :type dof:  int
    :return:    One of the 3 spin matrices for spin (dof -1) / 2
    :rtype:     dof x dof numpy array(complex)
    """
    j = (dof - 1) / 2
    if j == 0:
        return 0
    m_vals = np.linspace(j, -j, num = dof)
    if ind == 2:
        ret = np.zeros([dof, dof], dtype=complex)
        for i in range(dof):
            ret[i][i] = m_vals[i] / j
        return ret
    ladd_plus = np.zeros([dof, dof], dtype=complex)
    ladd_minus = np.zeros([dof, dof], dtype=complex)
    for i in range(dof - 1):
        m = m_vals[i + 1]
        ladd_plus[i][i+1] = np.sqrt(j * (j+1) - m * (m + 1))
        m = m_vals[i]
        ladd_minus[i + 1][i] = np.sqrt(j * (j + 1) - m * (m - 1))
    if ind == 0:
        return (ladd_plus + ladd_minus) / (2 * j)
    if ind == 1:
        return -1j * (ladd_plus - ladd_minus) / ( 2 * j)
    else:
        print('Index must be 0, 1, or 2')
        return None

def dens_spi_iso(site, myL, spI, isoI, spin=2, isospin=2):
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
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int    
    :return:        density operator as a sparse matrix
    :rtype:         scipy.sparse.coo_array(complex)
    """
    dim = myL ** 3 * spin * isospin
    ret = np.zeros([dim, dim],dtype=complex)

    sigma_s = get_spin_op(spI, spin)
    tau_i = get_spin_op(isoI, isospin)
    tensProd = np.tensordot(sigma_s, tau_i, axes=0)
    
    for ind, ele in np.ndenumerate(tensProd):
        if ele == 0:
            continue
        sz1, sz2, tz1, tz2 = ind
        pos1 = lat.state2index([site[0], site[1], site[2], isospin - tz1 - 1,
                                spin - sz1 - 1], myL, spin, isospin)
        pos2 =lat.state2index([site[0], site[1], site[2], isospin - tz2 - 1, 
                               spin  - sz2 - 1], myL, spin, isospin)
        ret[pos1, pos2] = ele
    ret= sparse.coo_array(ret)
    return ret

def f_SS(myL, bpi, a_lat):
    """
    f_S'S function as defined in equation 16
    
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
    ft_fss = np.exp(-bpi * (q2 + m_pi ** 2)) / (q2 + m_pi**2)

    q = np.array([qx, qy, qz])

    fSS = np.zeros((3, 3, myL, myL, myL), dtype=complex)

    for s1, s2 in itertools.product(range(3), range(3)):
        fSS[s1, s2] = np.fft.ifftn(q[s1] * q[s2] * ft_fss)

    return fSS

def onePionEx(myL, bpi, a_lat, lattice, verbose = False, mult = 1, spin=2, isospin=2):
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
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int    
    :return:        list of lists [i, j, k, l, value] where i, j and k, l are indices of two particles
                    in the single-particle basis, and value is the value of the matrix element <ij||kl>
                    in MeV. 
                    All elements have i<j and k<l
    :rtype:         list[(int, int, int, int, float)]  
    """
    ret = []
    
    if verbose:
        print('Calculating f_SS...', end='')
    fSS = f_SS(myL, bpi, a_lat)
    scale = - (consts.g_A / (2.0 * a_lat * consts.f_pi)) ** 2 * mult / 2.0
    dim  = myL **3 * spin * isospin
    if verbose:
        print('Done\nCalculating Densities...', end='')
    dens = [None] * myL ** 3
    for site in lattice:
        rho_sp = [None] * 3
        for sp in range(3):
            rho_sp_iso = [None] * 3
            for iso in range(3):
                rho_sp_iso[iso] = dens_spi_iso(site, myL, sp, iso, spin, isospin)
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
                if sys.getsizeof(tmp_val) >= 1e8:
                    if verbose:
                        print(f'Compressing interaction...',end='')
                    sparse_full_ope += sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim * dim, dim * dim))
                    sparse_full_ope.sum_duplicates()
                    tmp_val = []
                    tmp_row = []
                    tmp_col = []
                    if verbose:
                        print('Compressed')
    if len(tmp_col) > 0:
        if verbose:
            print(f'Compressing interaction...',end='')
        sparse_full_ope += sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape = (dim * dim, dim * dim))
        sparse_full_ope.sum_duplicates()
        if verbose:
            print('Done')
    if verbose:
        print('Interaction Generated\nConverting to List...', end='')
    del(tmp_val)
    del(tmp_row)
    del(tmp_col)
    
    sparse_full_ope = sparse_full_ope.tocoo()
    for i, j, v in zip(sparse_full_ope.row, sparse_full_ope.col, sparse_full_ope.data):
        a = i % dim
        b = (i - a) // dim
        c = j % dim
        d = (j - c) // dim

        ret.append([a, b, c, d, v])
    if verbose:
        print('Done')

    return ret

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

def tKin(myL, Nk, a_lat, spin=2, isospin=2):
    """
    computes 1-body kinetic energy matrix elements.
    
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param Nk:      number of neighbors along each axis to use(1 for nearest-neighbor, 2 for
                    next-to-nearest-neighbor, etc)
    :type Nk:       int
    :param a_lat:   lattice spacing divided by hbar c
    :type a_lat:    float
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int    
    :return:        list of tuples [i, j, value] where i and j are indices in the single-particle 
                    basis, and value is the value of the matrix element Tij in MeV
    :rtype:         list[(int, int, float)]
    """
    h = -1.0 / 2.0 / (consts.mass*a_lat)

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
    ret = []
    for i in range(myL ** 3):
        for j in range(myL ** 3):
            val = KK[i][j]
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
                    ret.append([ind1, ind2, val / a_lat])
    return ret