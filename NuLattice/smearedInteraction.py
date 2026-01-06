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

def smearMat(myL, sL, sNL):
    """
    gets the smearing matrix to get the local and nonlocal creation amd annihilation operators

    :param myL: number of lattice sites in each direction
    :type myL:  int
    :param sL:  strength of the local interaction
    :type sL:   float
    :param sNL:  strength of the nonlocal interaction
    :type sNL:   float
    """

    nn = (np.linspace(0,myL**3-1, num=myL**3)).astype(int)
    nx = nn % myL
    ny = ((nn - nx) / myL) % myL
    nz = (nn-nx-ny*myL) / myL ** 2
    nxp = (((nx+1) % myL) + ny*myL + nz*myL**2).astype(int)
    nxm = (((nx-1) % myL) + ny*myL + nz*myL**2).astype(int)
    nyp = (nx + ((ny+1) % myL)*myL + nz*myL**2).astype(int)
    nym = (nx + ((ny-1) % myL)*myL + nz*myL**2).astype(int)
    nzp = (nx + ny*myL + ((nz+1) % myL)*myL**2).astype(int)
    nzm = (nx + ny*myL + ((nz-1) % myL)*myL**2).astype(int)
    smear = np.zeros([myL**3, myL**3])
    for i in range(myL**3):
        smear[nn[i]][nxp[i]] += 1
        smear[nn[i]][nxm[i]] += 1
        smear[nn[i]][nyp[i]] += 1
        smear[nn[i]][nym[i]] += 1
        smear[nn[i]][nzp[i]] += 1
        smear[nn[i]][nzm[i]] += 1
        
    id = np.identity(myL**3)
    locMat = sL*smear + id
    nonLocMat = sNL*smear + id
    return locMat, nonLocMat

def smearedInteract(myL, c0, sL, sNL):
    """
    gets the smeared interaction as a function of two positions

    :param myL: number of lattice sites in each direction
    :type myL:  int
    :param c0:  strength of short-range interactions
    :type c0:   float
    :param sL:  strength of the local interaction
    :type sL:   float
    :param sNL:  strength of the nonlocal interaction
    :type sNL:   float
    """

    local, nonLocal = smearMat(myL, sL, sNL)
    locOrigin = np.zeros([myL**3])
    locOrigin[0] = 1.0

    fSL = np.diag(contract('ij, jk, k -> i', local, local, locOrigin))
    nonLocSq = contract('ij, jk -> ik', nonLocal, nonLocal)
    return c0 * contract('ij, jk, kl -> il', nonLocSq, fSL, nonLocSq)

def f_SL(site1, site2, myL, sL):
    if site1 == site2:
        return 1
    i1, j1, k1 = site1
    i2, j2, k2 = site2
    dist = ((i1 - i2 + myL // 2) % myL - myL // 2) ** 2 + ((j1 - j2 + myL // 2) % myL - myL // 2) ** 2 + ((k1 - k2 + myL // 2) % myL - myL // 2) ** 2
    if dist== 1:
        return sL
    return 0
    
def indConvXYZ(ind, myL):
    x = ind % myL
    y = ((ind - x) // myL) % myL
    z = ((ind - x) // myL - y) // myL
    return x, y, z

def periodicDistSq(site1, site2, myL):
    i1, j1, k1 = site1
    i2, j2, k2 = site2
    return ((i1 - i2 + myL // 2) % myL - myL // 2) ** 2 + ((j1 - j2 + myL // 2) % myL - myL // 2) ** 2 + ((k1 - k2 + myL // 2) % myL - myL // 2) ** 2

def potMat(lattice, myL, sL, sNL, c0):
    ret = np.zeros([myL ** 3, myL ** 3, myL ** 3, myL ** 3])
    for site1, site2, site3 in itertools.product(lattice, lattice, lattice):
        f_sl1 = f_SL(site1, site2, myL, sL)
        if f_sl1 == 0:
            continue
        f_sl2 = f_SL(site1, site3, myL, sL)
        if f_sl2 == 0:
            continue
        val = f_sl1 * f_sl2 * c0 / 4
        
        pos1 = site2[0] + site2[1] * myL + site2[2] * myL ** 2
        pos2 = site3[0] + site3[1] * myL + site3[2] * myL ** 2
        rx1 = lat.right(site2[0], myL) + site2[1] * myL + site2[2] * myL ** 2
        rx2 = lat.right(site3[0], myL) + site3[1] * myL + site3[2] * myL ** 2
        ry1 = lat.right(site2[1], myL) * myL + site2[0] + site2[2] * myL ** 2
        ry2 = lat.right(site3[1], myL) * myL + site3[0] + site3[2] * myL ** 2
        rz1 = lat.right(site2[2], myL) * myL ** 2 + site2[0] + site2[1] * myL
        rz2 = lat.right(site3[2], myL) * myL ** 2 + site3[0] + site3[1] * myL
        lx1 = lat.left(site2[0], myL)  + site2[1] * myL + site2[2] * myL ** 2
        lx2 = lat.left(site3[0], myL)  + site3[1] * myL + site3[2] * myL ** 2
        ly1 = lat.left(site2[1], myL)  * myL + site2[0] + site2[2] * myL ** 2
        ly2 = lat.left(site3[1], myL)  * myL + site3[0] + site3[2] * myL ** 2
        lz1 = lat.left(site2[2], myL)  * myL ** 2 + site2[0] + site2[1] * myL
        lz2 = lat.left(site3[2], myL)  * myL ** 2 + site3[0] + site3[1] * myL

        pos1_lst = [pos1, rx1, ry1, rz1, lx1, ly1, lz1]
        pos2_lst = [pos2, rx2, ry2, rz2, lx2, ly2, lz2]
        for i, j, k, l in itertools.product(pos1_lst, pos2_lst, pos1_lst, pos2_lst):
            count = 0
            if i != pos1:
                count +=1
            if j != pos2:
                count += 1
            if k != pos1:
                count += 1
            if l != pos2:
                count += 1
            ret[i,j,k,l] += val * (sNL ** count)
            ret[j,i,l,k] += val * (sNL ** count)
            # if i != j:
            ret[j,i,k,l] -= val * (sNL ** count)
            # if k != l:
            ret[i,j,l,k] -= val * (sNL ** count)
    return ret


def nonLocOp(site, myL, sNL, sz, tz,spin=2,isospin=2):
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

def oneBodOp(site, myL, sNL, spin=2, isospin=2):
    """
    Generates one body operator as defined in equation 9
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
    """
    ret = []


    #list of density matrices
    rhos = []
    
    for site in lattice:
        rhos.append(oneBodOp(site, myL, sNL, spin, isospin))
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
    count = 0
    tmp_col = []
    tmp_row = []
    tmp_val = []
    for optempn in rho_fsl:
        
        for a, c, v in zip(optempn.row, optempn.col, optempn.data):
            for b, d, w in zip(optempn.row, optempn.col, optempn.data):
                
                matele = c0 * v * w
                if a < c and b > d:
                    tmp_col.append(a + c * dim)
                    tmp_row.append(b + c * dim)
                    tmp_val.append(matele)
                if c < a and b < d:
                    tmp_col.append(c + a * dim)
                    tmp_row.append(b + c * dim)
                    tmp_val.append(-matele)
                # if a < b and c < d:
                #     tmp_col.append(a + b * dim)
                #     tmp_row.append(d + c * dim)
                #     tmp_val.append(matele)
                #     # indI = a + b * dim
                #     # indJ = d + c * dim
                #     # sparse_full_int[indI, indJ] += matele
                #     # ret.append([a, b, d, c, - matele])
                # if a < b and c > d:
                #     tmp_col.append(a + b * dim)
                #     tmp_row.append(c + d * dim)
                #     tmp_val.append(-matele)
                #     # indI = a + b * dim
                #     # indJ = c + d * dim
                #     # sparse_full_int[indI, indJ] -= matele
                #     # ret.append([a, b, c, d, matele])
                # elif a > b and c < d:
                #     tmp_col.append(b + a * dim)
                #     tmp_row.append(d + c * dim)
                #     tmp_val.append(-matele)
                #     # indI = b + a * dim
                #     # indJ = d + c * dim
                #     # sparse_full_int[indI, indJ] -= matele
                #     # ret.append([b, a, d, c, matele])
                # elif a > b and c > d:
                #     tmp_col.append(b + a * dim)
                #     tmp_row.append(c + d * dim)
                #     tmp_val.append(matele)
                #     # indI = b + a * dim
                #     # indJ = c + d * dim
                #     # sparse_full_int[indI, indJ] += matele
                #     # ret.append([b, a, c, d, - matele])
                if sys.getsizeof(tmp_val) >= 1e8:
                    count += 1
                    print(f'Compressing {count}')
                    sparse_full_int += sparse.coo_array((tmp_val, (tmp_row, tmp_col)), shape = (dim * dim, dim * dim))
                    sparse_full_int.sum_duplicates()
                    tmp_val = []
                    tmp_row = []
                    tmp_col = []
    if len(tmp_col) > 0:
        sparse_full_int += sparse.coo_array((tmp_val, (tmp_row, tmp_col)), shape = (dim * dim, dim * dim))
        sparse_full_int.sum_duplicates()
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
            
    print('Summing Duplicates')
    #Sum all of the entries that are at the same point
    my_basis = lat.get_sp_basis(myL)
    ret = fbd.add_2body_ops([ret, ret], my_basis,[0.5, 0.5])
    print('Full Interaction Generated')

    return ret

def pot_mat_to_tbme(mat, myL, spin=2, isospin=2):
    ret = []
    nzInds = np.nonzero(mat)
    nzVals = mat[nzInds]
    nzLst = np.column_stack((*nzInds, nzVals))
    for sz1, tz1, sz2, tz2 in itertools.product(range(spin), range(isospin), range(spin), range(isospin)):
        # for sz3, tz3, sz4, tz4 in itertools.product(range(spin), range(isospin), range(spin), range(isospin)):
            sz3, sz4, tz3, tz4 = sz1, sz2, tz1, tz2
            # if tz1 + tz2 != tz3 + tz4: #Tz is not conserved
            #     continue
            # if sz1+sz2 != sz3+sz4: #Sz is not conserved
            #     continue
            for indx in nzLst:
                i, j, k, l, val = indx
                xi, yi, zi = indConvXYZ(i, myL)
                xj, yj, zj = indConvXYZ(j, myL)
                xk, yk, zk = indConvXYZ(k, myL)
                xl, yl, zl = indConvXYZ(l, myL)
                indi = lat.state2index([xi, yi, zi, sz1, tz1], myL)
                indj = lat.state2index([xj, yj, zj, sz2, tz2], myL)
                indk = lat.state2index([xk, yk, zk, sz3, tz3], myL)
                indl = lat.state2index([xl, yl, zl, sz4, tz4], myL)
                # if indi == indj and tz1 == tz2 and sz1 == sz2:
                #     continue
                # if indk == indl and tz3 == tz4 and sz3 == sz4: #not asymetric under exchange
                #     continue
                if indi < indj and indl > indk:
                    ret.append([indi, indj, indk, indl, val])        
    return ret

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

def interact(interMat, lattice, myL, spin=2, isospin=2):
    """
    computes matrix elements for 2-body onsite contacts
    
    :param interMat: matrix that takes two lattice positions and returns the strength of the interation
    :type interMat: 2D numpy array with dimension myL ** 3 by myL ** 3
    :param lattice: list of lattice sites returned by get_lattice
    :type lattice:  list[(int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int    
    :return:    list of lists [i, j, k, l, value] where i, j and k, l are indices of two particles
                in the single-particle basis, and value is the value of the matrix element <ij||kl>. 
                All elements have i<j and k<l
    :rtype:         list[(int, int, int, int, float)]
    """
    matele = []
    ind_set = set([])
    for site1 in lattice:
        for tz1 in range(isospin):
            for sz1 in range(spin):
                stat1 = copy.deepcopy(site1)
                stat1.append(tz1)
                stat1.append(sz1)
                indx1 = lat.state2index(stat1, myL=myL, spin=spin, isospin=isospin)
                for site2 in lattice:
                    for tz2 in range(isospin):
                        for sz2 in range(spin):
                            if site1 == site2 and tz1 == tz2 and sz1 == sz2: #not asymetric under exchange
                                continue
                            stat2 = copy.deepcopy(site2)
                            stat2.append(tz2)
                            stat2.append(sz2)
                            indx2 = lat.state2index(stat2, myL=myL, spin=spin, isospin=isospin)
                            if indx2 <= indx1: #we only keep properly ordered two-body states
                                continue
                            # for tz3 in range(isospin):
                            #     for sz3 in range(spin):
                            sz3, sz4, tz3, tz4 = sz1, sz2, tz1, tz2
                            stat3 = copy.deepcopy(site1)
                            stat3.append(tz3)
                            stat3.append(sz3)
                            indx3 = lat.state2index(stat3, myL=myL, spin=spin, isospin=isospin)
                                    # for tz4 in range(isospin):
                            if tz1 + tz2 != tz3 + tz4: #Tz is not conserved
                                continue
                            # for sz4 in range(spin):
                            if sz1+sz2 != sz3+sz4: #Sz is not conserved
                                continue
                            if site1 == site2 and tz3 == tz4 and sz3 == sz4: #not asymetric under exchange
                                continue
                            stat4 = copy.deepcopy(site2)
                            stat4.append(tz4)
                            stat4.append(sz4)
                            indx4 = lat.state2index(stat4, myL=myL, spin=spin, isospin=isospin)
                            factor = 1
                            pos1 = site1[0] + site1[1]* myL + site1[2] * myL ** 2
                            pos2 = site2[0] + site2[1]* myL + site2[2] * myL ** 2
                            if indx4 <= indx3:
                                continue
                                                                        
                            # if (indx1, indx2, indx3, indx4) not in ind_set:
                            #     ind_set.add((indx1, indx2, indx3, indx4))
                            matele.append([indx1, indx2, indx3, indx4, factor * interMat[pos1, pos2]])  
    return matele

def tKin(lattice, myL, unit=1, spin=2, isospin=2):
    """
    computes 1-body kinetic energy matrix elements. Really: the negative laplacian 

    :param lattice: list of lattice sites returned by get_lattice
    :type lattice:  list[(int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int    
    :param unit:    Optional; unit of the kinetic energy, can also give a general scale factor 
                    to match the units of another calculation
    :type unit:     float
    :return:    list of tuples [i, j, value] where i and j are indices in the single-particle 
                basis, and value is the value of the matrix element Tij
    :rtype:     list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        #diagonal element from each direction
        val = (15.0 / 2.0) * unit
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx1, val])

        rx = lat.right(i, myL=myL)
        rx2 = lat.right(rx, myL=myL)
        ry = lat.right(j, myL=myL)
        ry2 = lat.right(ry, myL=myL)
        rz = lat.right(k, myL=myL)
        rz2 = lat.right(rz, myL=myL)
        val1 = -(4.0 / 3.0) * unit
        val2 = (1.0 / 12.0) * unit
        for tz in range(isospin):
            for sz in range(spin):
                #hop one in x
                state1 = [i, j, k, tz, sz]
                state2 = [rx, j, k, tz, sz]
                indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val1])
                mat.append([indx2, indx1, val1]) #adds a hop-to-the left matrix element

                #hop two in x
                state1 = [i, j, k, tz, sz]
                state2 = [rx2, j, k, tz, sz]
                indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val2])
                mat.append([indx2, indx1, val2]) #adds a hop-to-the left matrix element

                #hop one in y
                state1 = [i, j, k, tz, sz]
                state2 = [i, ry, k, tz, sz]
                indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val1])
                mat.append([indx2, indx1, val1]) #adds a hop-to-the left matrix element

                #hop two in y
                state1 = [i, j, k, tz, sz]
                state2 = [i, ry2, k, tz, sz]
                indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val2])
                mat.append([indx2, indx1, val2]) #adds a hop-to-the left matrix element

                #hop one in z
                state1 = [i, j, k, tz, sz]
                state2 = [i, j, rz, tz, sz]
                indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val1])
                mat.append([indx2, indx1, val1]) #adds a hop-to-the left matrix element

                #hop two in z
                state1 = [i, j, k, tz, sz]
                state2 = [i, j, rz2, tz, sz]
                indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val2])
                mat.append([indx2, indx1, val2]) #adds a hop-to-the left matrix element
    return mat

def get_full_int(myL, bpi, c0, sL, sNL, a, at, method=1, nk = 2, spin = 2, isospin = 2):
    """
    Takes the parameters needed for the smeared interaction, and returns the one and two body matrix elements

    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param bpi: parameter to remove short-distance lattice artifacts
    :type bpi: float
    :param c0:  strength of short-range interactions
    :type c0:   float
    :param sL:  strength of the local interaction
    :type sL:   float
    :param sNL:  strength of the nonlocal interaction
    :type sNL:   float
    :param a_lat: lattice spacing in fm
    :type a_lat: float
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int    
    """
    lattice = lat.get_lattice(myL)
    unit = lat.phys_unit(consts.hbarc * a)
    if method == 1:
        kin = tKin(lattice, myL, unit * a)    
        kinMat = tKinListToMat(kin, myL, spin, isospin)
    else:
        kinMat = tKin2(myL, nk, a)
        kin = tKinMatToList(kinMat, myL, spin, isospin)
    id = np.identity(myL**3)
    trMat = (id - at * kinMat) @ (id - at * kinMat) - at * np.real((smearedInteract(myL, c0, sL, sNL)) + onePionEx(myL, bpi, 0, a))
    interMat = trMat / at - kinMat
    full_int = interact(interMat, lattice, myL, spin, isospin)
    return kin, full_int, trMat

def tKin2(myL, Nk, a):
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
    return KK

def tKinListToMat(tLst, myL, spin=2, isospin=2):
    ret = np.zeros([myL **3, myL ** 3])
    for i in tLst:
        ind1, sz1, tz1 = indConv(i[0], myL, spin, isospin)
        ind2, sz2, tz2 = indConv(i[1], myL, spin, isospin)
        if sz1 + tz1 + sz2 + tz2 == 0:
            ret[ind1, ind2] += i[2] 
    return ret

def tKinMatToList(tMat, myL, spin = 2, isospin =2):
    ret = []
    for i in range(myL ** 3):
        for j in range(myL ** 3):
            val = tMat[i][j]
            if val == 0:
                continue
            indx1, indy1, indz1 = indConvXYZ(i, myL)
            indx2, indy2, indz2 = indConvXYZ(j, myL)
            for tz in range(isospin):
                for sz in range(spin):
                    state1 = [indx1, indy1, indz1, tz, sz]
                    ind1 = lat.state2index(state1, myL, spin, isospin)
                    state2 = [indx2, indy2, indz2, tz, sz]
                    ind2 = lat.state2index(state2, myL, spin, isospin)
                    ret.append([ind1, ind2, val])
    return ret

def indConv(ind, myL, spin=2, isospin=2):
    sz = ind % spin
    tz = ((ind - sz) // spin) % isospin
    indx = (ind - sz - spin * tz) // (spin * isospin)
    k = indx % myL
    j = ((indx - k)// myL) % myL
    i = ((indx - k) // myL - j) // myL
    return i + j * myL + k * myL ** 2, sz, tz

def potLst(lattice, myL, sL, sNL, c0, my_basis, spin = 2, isospin = 2):
    ret = []
    for site1, site2, site3 in itertools.product(lattice, lattice, lattice):
        f_sl1 = f_SL(site1, site2, myL, sL)
        if f_sl1 == 0:
            continue
        f_sl2 = f_SL(site1, site3, myL, sL)
        if f_sl2 == 0:
            continue
        val = f_sl1 * f_sl2 * c0 / 2
        
        rx1 = [lat.right(site2[0], myL), site2[1], site2[2]]
        rx2 = [lat.right(site3[0], myL), site3[1], site3[2]]
        lx1 = [lat.left(site2[0], myL), site2[1], site2[2]]
        lx2 = [lat.left(site3[0], myL), site3[1], site3[2]]

        ry1 = [site2[0], lat.right(site2[1], myL), site2[2]]
        ry2 = [site3[0], lat.right(site3[1], myL), site3[2]]
        ly1 = [site2[0], lat.left(site2[1], myL), site2[2]]
        ly2 = [site3[0], lat.left(site3[1], myL), site3[2]]

        rz1 = [site2[0], site2[1], lat.right(site2[2], myL)]
        rz2 = [site3[0], site2[1], lat.right(site3[2], myL)]
        lz1 = [site2[0], site2[1], lat.left(site2[2], myL)]
        lz2 = [site3[0], site2[1], lat.left(site3[2], myL)]

        pos1_lst = [site2, rx1, ry1, rz1, lx1, ly1, lz1]
        pos2_lst = [site3, rx2, ry2, rz2, lx2, ly2, lz2]
        for sz1, tz1, sz2, tz2 in itertools.product(range(spin), range(isospin), range(spin), range(isospin)):
            for sz3, tz3, sz4, tz4 in itertools.product(range(spin), range(isospin), range(spin), range(isospin)):
                if tz1 + tz2 != tz3 + tz4: #Tz is not conserved
                    continue
                if sz1+sz2 != sz3+sz4: #Sz is not conserved
                    continue
                for i, j, k, l in itertools.product(pos1_lst, pos2_lst, pos1_lst, pos2_lst):
                    count = 0
                    if i != site2:
                        count +=1
                    if j != site3:
                        count += 1
                    if k != site2:
                        count += 1
                    if l != site3:
                        count += 1
                    indi = lat.state2index([i[0], i[1], i[2], sz1, tz1], myL)
                    indj = lat.state2index([j[0], j[1], j[2], sz2, tz2], myL)
                    indk = lat.state2index([k[0], k[1], k[2], sz3, tz3], myL)
                    indl = lat.state2index([l[0], l[1], l[2], sz4, tz4], myL)
                    if indi == indj and tz1 == tz2 and sz1 == sz2:
                        continue
                    if indk == indl and tz3 == tz4 and sz3 == sz4: #not asymetric under exchange
                        continue
                    if indi < indj and indl > indk: 
                        ret.append([indi, indj, indk, indl, val * (sNL ** count)])
    return fbd.add_2body_ops([ret,ret], my_basis, weights=[0.5,0.5])
