import numpy as np
from opt_einsum import contract
import copy
import NuLattice.lattice as lat
import NuLattice.constants as consts

def smearMat(myL, sL, sNL):
    """
    gets the smearing matrix to get the local and nonlocal creatopm amd annihilation operators

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
    return c0 / 2 * contract('ij, jk, kl -> il', nonLocSq, fSL, nonLocSq)

def onePionEx(myL, bpi, spin, a_lat):
    """
    computes the potential for one pion exchange

    :param myL: number of lattice sites in each direction
    :type myL:  int
    :param bpi: parameter to remove short-distance lattice artifacts
    :type bpi: float
    :param spin: Not sure why this is specified tbh
    :type spin: int
    :param a_lat: lattice spacing in fm
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
                            if tz1 == tz2 and sz1 == sz2: #not asymetric under exchange
                                continue
                            stat2 = copy.deepcopy(site2)
                            stat2.append(tz2)
                            stat2.append(sz2)
                            indx2 = lat.state2index(stat2, myL=myL, spin=spin, isospin=isospin)
                            if indx2 <= indx1: #we only keep properly ordered two-body states
                                continue
                            for tz3 in range(isospin):
                                for sz3 in range(spin):
                                    stat3 = copy.deepcopy(site2)
                                    stat3.append(tz3)
                                    stat3.append(sz3)
                                    indx3 = lat.state2index(stat3, myL=myL, spin=spin, isospin=isospin)
                                    for tz4 in range(isospin):
                                        if tz1 + tz2 != tz3 + tz4: #Tz is not conserved
                                            continue
                                        for sz4 in range(spin):
                                            if sz1+sz2 != sz3+sz4: #Sz is not conserved
                                                continue
                                            if tz3 == tz4 and sz3 == sz4: #not asymetric under exchange
                                                continue
                                            stat4 = copy.deepcopy(site1)
                                            stat4.append(tz4)
                                            stat4.append(sz4)
                                            indx4 = lat.state2index(stat4, myL=myL, spin=spin, isospin=isospin)
                                            if indx4 <= indx3:
                                                continue
                                            pos1 = site1[0] + site1[1]* myL + site1[2] * myL ** 2
                                            pos2 = site2[0] + site2[1]* myL + site2[2] * myL ** 2
                                            matele.append([indx1, indx2, indx3, indx4, interMat[pos1, pos2]])
    return matele

def Tkin(lattice, myL, a_lat, spin=2, isospin=2):
    """
    computes 1-body kinetic energy matrix elements. Really: the negative dimensionless laplacian 

    :param lattice: list of lattice sites returned by get_lattice
    :type lattice:  list[(int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int    
    :return:    list of tuples [i, j, value] where i and j are indices in the single-particle 
                basis, and value is the value of the matrix element Tij
    :rtype:     list[(int, int, float)]
    """
    mat = []
    unit = lat.phys_unit(a_lat)
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        #diagonal element from each direction
        val = 15.0 / 2.0 * unit
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
        val1 = -4.0 / 3.0 * unit
        val2 = 1.0 / 12.0 * unit
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

def get_full_int(myL, bpi, c0, sL, sNL, a_lat, spin = 2, isospin = 2):
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
    a =  a_lat / consts.hbarc
    kin = Tkin(lattice, myL, lat.phys_unit(a_lat) / a, spin, isospin)
    interMat = (onePionEx(myL, bpi, 0, a) + smearedInteract(myL, c0, sL, sNL))
    full_int = interact(interMat, lattice, myL, spin, isospin)
    return kin, np.real(full_int)
