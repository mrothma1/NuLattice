import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".."))
import NuLattice.lattice as lat
import NuLattice.FCI.few_body_diagonalization as fbd
from scipy.sparse.linalg import eigsh as arpack_eigsh

# Initialize lattice
# Note: on a laptop, an estimate is that H2 can be solved on L=10, He3 and H3 on L=6, and He4 on L=4 
thisL = 3
a_lat = 2.0 # lattice spacing in fm

phys_unit = lat.phys_unit(a_lat)

my_basis = lat.get_sp_basis(thisL)
nstat =  len(my_basis)
print("number of single-particle states =", nstat)
lattice = lat.get_lattice(thisL)
nsite = len(lattice)
print("number of lattice sites =", nsite)

# Compute operators for kinetic energy, two-body contacts, and three-body contact

vT1=-8.0  #S-wave triplet contact
vS1=-8.0  #S-wave triplet contact
cE = 5.5  #three-body contact

myTkin=lat.Tkin(lattice, thisL)
print("number of matrix elements from kinetic energy", len(myTkin))

mycontact=lat.contacts(vT1, vS1, lattice, thisL)
print("number of matrix elements from two-body contacts", len(mycontact))

my3body=lat.NNNcontact(cE, lattice, thisL)
print("number of matrix elements from three-body contacts", len(my3body))

# Compute the deuteron
print("Computing deuteron")
# additive quantum numbers
numpart=2 # number of nucleons
tz = 0    # twice the value of isospin projection
sz = 2    # twice the value of spin projection

# get two-body basis as a dictionary for lookup
H2_lookup = fbd.get_many_body_states(my_basis, numpart, total_tz=tz, total_sz=sz)
print("matrix dimension:", len(H2_lookup))

# make scipy.sparse.csr_matrix for kinetic energy 
T2_csr_mat = fbd.get_csr_matrix_scalar_op(H2_lookup, myTkin, nstat)
print("kinetic energy done")

# make scipy.sparse.csr_matrix for 2-body interactions 
V2_csr_mat = fbd.get_csr_matrix_scalar_op(H2_lookup, mycontact, nstat)
print("2-body interaction done")

# add all into Hamiltonian
H2_csr_mat = T2_csr_mat + V2_csr_mat

# compute lowest eigenvalue(s)
k_eig=10  # number of eigenvalues
vals, vecs = arpack_eigsh(H2_csr_mat, k=k_eig, which='SA')
print("Energies (MeV):", vals*phys_unit)

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
k_eig=10  # number of eigenvalues
vals, vecs = arpack_eigsh(H3_csr_mat, k=k_eig, which='SA')
print("Energies (MeV):", vals*phys_unit)

# Compute He4
print("Computing 4He")
numpart=4
tz = 0 # twice the value
sz = 0 # twice the value

# get four-body basis as a dictionary for lookup
He4_lookup = fbd.get_many_body_states(my_basis, numpart, total_tz=tz, total_sz=sz)
print("matrix dimension:", len(He4_lookup))

# make scipy.sparse.csr_matrix for kinetic energy 
T4_csr_mat = fbd.get_csr_matrix_scalar_op(He4_lookup, myTkin, nstat)
print("kinetic energy done")

# make scipy.sparse.csr_matrix for 2-body interaction 
V4_csr_mat = fbd.get_csr_matrix_scalar_op(He4_lookup, mycontact, nstat)
print("2-body interaction done")

# make scipy.sparse.csr_matrix for 3-body interaction 
W4_csr_mat = fbd.get_csr_matrix_scalar_op(He4_lookup, my3body, nstat)
print("3-body interaction done")

# add all into Hamiltonian
H4_csr_mat = T4_csr_mat + V4_csr_mat + W4_csr_mat

# compute lowest eigenvalue(s)
k_eig=2  # number of eigenvalues
vals, vecs = arpack_eigsh(H4_csr_mat, k=k_eig, which='SA')
print("Energies (MeV):", vals*phys_unit)
