import NuLattice.smearedInteraction as smearI
import NuLattice.FCI.few_body_diagonalization as fbd
import NuLattice.lattice as lat
from scipy.sparse.linalg import eigsh as arpack_eigsh
import NuLattice.constants as consts

sL = 0.5
sNL = 0.061
a = 1.32 / consts.hbarc
c2 = -3.41e-7 / a ** 2
c3 = -1.4e-14 / a ** 5
thisL = 3
lattice = lat.get_lattice(thisL)
my_basis = lat.get_sp_basis(thisL)
nstat =  len(my_basis)

myTkin=smearI.tKin(thisL, 3, a)
print("number of matrix elements from kinetic energy", len(myTkin))

mycontact=smearI.shortRangeV_2body(lattice, thisL, sL, sNL, c2, a, verbose=True)
mycontact = smearI.sparse_to_list_2body(mycontact, thisL)
print("number of matrix elements from two-body contacts", len(mycontact))

my3body=smearI.shortRangeV_3body(lattice, thisL, sL, sNL, c3, a,verbose=True, min_val = sNL ** 2 * sL, max_threads=30)
my3body=smearI.sparse_to_list_3body(my3body, thisL)
print("number of matrix elements from three-body contacts", len(my3body))

# Compute He3
print("Computing 3He")
numpart=3
tz = -1 # twice the value
sz = -1 # twice the value

# get three-body basis as a dictionary for lookup
He3_lookup = fbd.get_many_body_states(my_basis, numpart, total_tz=tz, total_sz=sz)
print("matrix dimension:", len(He3_lookup))

# make scipy.sparse.csr_matrix for kinetic energy 
T3_csr_mat = fbd.get_csr_matrix_scalar_op(He3_lookup, myTkin, nstat)
print("kinetic energy done")

# make scipy.sparse.csr_matrix for 2-body interaction 
V3_csr_mat = fbd.get_csr_matrix_scalar_op(He3_lookup, mycontact, nstat)
print("2-body interaction done")

# make scipy.sparse.csr_matrix for 3-body interaction 
W3_csr_mat = fbd.get_csr_matrix_scalar_op(He3_lookup, my3body, nstat)
print("3-body interaction done")

# add all into Hamiltonian
H3_csr_mat = T3_csr_mat + V3_csr_mat + W3_csr_mat

# compute lowest eigenvalue(s)
k_eig=2  # number of eigenvalues
vals, vecs = arpack_eigsh(H3_csr_mat, k=k_eig, which='SA')
print("Energies (MeV):", vals)