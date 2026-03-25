import NuLattice.smearedInteraction as smearI
import NuLattice.FCI.few_body_diagonalization as fbd
import NuLattice.lattice as lat
from scipy.sparse.linalg import eigsh as arpack_eigsh
import NuLattice.constants as consts

sL = 0.08
sNL = 0.08
a = 1 / 100
c2 = -0.185
bpi = 0.7
for thisL in [4, 5, 6]:
    lattice = lat.get_lattice(thisL)
    my_basis = lat.get_sp_basis(thisL)
    nstat =  len(my_basis)

    myTkin=smearI.tKin(thisL, 3, a)
    print("number of matrix elements from kinetic energy", len(myTkin))

    mycontact=smearI.shortRangeV_2body(lattice, thisL, sL, sNL, c2, a, verbose=True)
    # myOPE = smearI.onePionEx(thisL, bpi, a, lattice,max_mem=0)
    mycontact = smearI.sparse_to_list_2body(mycontact, thisL)
    print("number of matrix elements from two-body contacts", len(mycontact))

    # Compute 2H
    print("Computing 2H")
    # additive quantum numbers
    numpart=2 # number of nucleons
    tz = 0    # twice the value of isospin projection
    sz = 2    # twice the value of spin projection

    # get two-body basis as a dictionary for lookup
    H2_lookup = fbd.get_many_body_states(my_basis, numpart, total_tz=tz, total_sz=None)
    print("matrix dimension:", len(H2_lookup))

    # make scipy.sparse.csr_matrix for kinetic energy 
    T3_csr_mat = fbd.get_csr_matrix_scalar_op(H2_lookup, myTkin, nstat)
    print("kinetic energy done")

    # make scipy.sparse.csr_matrix for 2-body interaction 
    V3_csr_mat = fbd.get_csr_matrix_scalar_op(H2_lookup, mycontact, nstat)
    print("2-body interaction done")

    # add all into Hamiltonian
    H3_csr_mat = T3_csr_mat + V3_csr_mat

    # compute lowest eigenvalue(s)
    k_eig=2  # number of eigenvalues
    vals, vecs = arpack_eigsh(H3_csr_mat, k=k_eig, which='SA')
    print("Energies (MeV):", vals[0])