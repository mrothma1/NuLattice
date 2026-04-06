import NuLattice.one_body_operators as obops
import NuLattice.two_body_operators as tbops
import NuLattice.lattice as lat
import NuLattice.FCI.few_body_diagonalization as fbd
from scipy.sparse.linalg import eigsh as arpack_eigsh
import NuLattice.references as ref
import numpy as np
import NuLattice.HF.hartree_fock as hf
if __name__ == '__main__':
    thisL = 7
    a = 1.0 / 100.0
    lattice = lat.get_lattice(thisL)

    myTkin=obops.tKin(thisL, 3, a)
    print("number of matrix elements from kinetic energy", len(myTkin))

    bpi = 0.7
    verbose = False
    v_OPE = tbops.onePionEx(thisL, bpi, a, lattice, verbose=verbose)
    # v_coulomb = tbops.coulomb_pot(lattice, thisL, a, verbose=verbose, max_mem=1e8)
    #INTERACTION A
    sNL = 0.077
    cNL = -0.2268
    cINL = 0.02184

    v_NL=tbops.shortRangeV_2body(lattice, thisL, 0, sNL, cNL, a, verbose=verbose, max_mem=1e8)
    iso_ops = [2.0 * obops.list_to_sparse1b(obops.tau_x(lattice, thisL)), 
               2.0 * obops.list_to_sparse1b(obops.tau_y(lattice, thisL)), 
               2.0 * obops.list_to_sparse1b(obops.tau_z(lattice, thisL))]
    for tau_I in iso_ops:
        v_NL += tbops.shortRangeV_2body(lattice, thisL, 0, sNL, cINL, a, op1b=tau_I,verbose=verbose, max_mem=1e8)

    mycontactA = tbops.sparse_to_list_2body(v_NL+v_OPE, thisL)
    # mycontactA_C = tbops.sparse_to_list_2body(v_NL+v_OPE+v_coulomb, thisL)
    print("number of matrix elements from two-body contacts", len(mycontactA))

    #INTERACTION B
    sNL = 0.077
    cNL = -0.1171
    cINL = 0.02607
    sL = 0.81
    cL = -0.01013
    cSL = - cL / 3
    cIL = cSL
    cSIL = cSL

    v_NL=tbops.shortRangeV_2body(lattice, thisL, 0, sNL, cNL, a, verbose=verbose, max_mem=1e8)
    iso_ops = [2.0 * obops.list_to_sparse1b(obops.tau_x(lattice, thisL)), 
            2.0 * obops.list_to_sparse1b(obops.tau_y(lattice, thisL)), 
            2.0 * obops.list_to_sparse1b(obops.tau_z(lattice, thisL))]
    for tau_I in iso_ops:
        v_NL += tbops.shortRangeV_2body(lattice, thisL, 0, sNL, cINL, a, op1b=tau_I,verbose=verbose, max_mem=1e8)

    v_L = tbops.shortRangeV_2body(lattice, thisL, sL, 0, cL, a,verbose=verbose,max_mem=1e8)
    sp_ops = [  2.0 * obops.list_to_sparse1b(obops.spin_x(lattice, thisL)), 
                2.0 * obops.list_to_sparse1b(obops.spin_y(lattice, thisL)), 
                2.0 * obops.list_to_sparse1b(obops.spin_z(lattice, thisL))]
    for tau_I in iso_ops:
        v_L += tbops.shortRangeV_2body(lattice, thisL, sL, 0, cIL, a, op1b=tau_I, verbose=verbose,max_mem=1e8)
        for sigma_S in sp_ops:
            v_L += tbops.shortRangeV_2body(lattice, thisL, sL, 0, cSIL, a, op1b=sigma_S @ tau_I, verbose=verbose, max_mem=1e8)
    for sigma_S in sp_ops:
            v_L += tbops.shortRangeV_2body(lattice, thisL, sL, 0, cSL, a, op1b=sigma_S, verbose=verbose, max_mem=1e8)
    mycontactB = tbops.sparse_to_list_2body(v_L+v_NL+v_OPE, thisL)
    # mycontactB_C = tbops.sparse_to_list_2body(v_L+v_NL+v_OPE+v_coulomb, thisL)
    print("number of matrix elements from two-body contacts", len(mycontactB))

    # we compute helium-4
    my_ref = ref.ref_4He_gs
    my_basis = lat.get_sp_basis(thisL)
    nstat =  len(my_basis)
    hole = ref.reference_to_holes(my_ref,my_basis)
    hnum = len(hole)

    dens = hf.init_density(nstat,hole,dtype=complex)
    print("number of particles:", np.real_if_close(np.trace(dens)), "compare with", hnum)

    eps=1.e-8
    mix = 0.9
    max_iter=1000

    erg, trafo, conv = hf.solve_HF(myTkin, mycontactA, [], dens,
                                   mix=mix, eps=eps, max_iter=max_iter, verbose=verbose)

    if conv:
        print("Interaction A HF energy (MeV) = ", erg)
    else:
        print("HF did not converge")

    # erg, trafo, conv = hf.solve_HF(myTkin, mycontactA_C, [], dens,
    #                                mix=mix, eps=eps, max_iter=max_iter, verbose=verbose)

    # if conv:
    #     print("Interaction A w/Coulomb HF energy (MeV) = ", erg)
    # else:
    #     print("HF did not converge")

    erg, trafo, conv = hf.solve_HF(myTkin, mycontactB, [], dens,
                                   mix=mix, eps=eps, max_iter=max_iter, verbose=verbose)

    if conv:
        print("Interaction B HF energy (MeV) = ", erg)
    else:
        print("HF did not converge")


    # erg, trafo, conv = hf.solve_HF(myTkin, mycontactB_C, [], dens,
    #                             mix=mix, eps=eps, max_iter=max_iter, verbose=verbose)

    # if conv:
    #     print("Interaction B w/Coulomb HF energy (MeV) = ", erg)
    # else:
    #     print("HF did not converge")