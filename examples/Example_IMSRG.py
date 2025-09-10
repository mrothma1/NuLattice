# Copyright 2025 Matthias Heinz. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Script to solve the IMSRG(2) equations for He3."""
__authors__   =  ["Matthias Heinz"]
__credits__   =  ["Matthias Heinz"]
__copyright__ = "(c) Matthias Heinz"
__license__   = "BSD-3-Clause"
__date__      = "2025-09-03"

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".."))

import NuLattice.lattice as lat
import NuLattice.references as ref

from NuLattice.IMSRG import normal_ordering
from NuLattice.IMSRG import ode_solver

# Lattice params
L = 2
a_lat = 2.5
phys_unit = lat.phys_unit(a_lat)

# Lattice basis
basis = lat.get_sp_basis(L)
lattice = lat.get_lattice(L)

# Couplings
vT1 = -9.0
vS1 = vT1
D = 6.0

# Kinetic energy and potential matrix elements
kin = lat.Tkin(lattice, L)
contact_nn = lat.contacts(vT1, vS1, lattice, L)
contact_3n = lat.NNNcontact(D, lattice, L)

# Reference state and occupations
he3_ref = ref.ref_3He_gs
occs = normal_ordering.create_occupations(basis, he3_ref)

# Normal ordered Hamiltonian
e0, f, gamma = normal_ordering.compute_normal_ordered_hamiltonian_no2b(
    occs, kin, contact_nn, contact_3n
)

# IMSRG(2) solution
e_imsrg, integration_data = ode_solver.solve_imsrg2(occs, e0, f, gamma, s_max=40, eta_criterion=1e-3)

print("E_IMSRG = {:>12.5f} (lattice units), {:>13.4f} MeV".format(e_imsrg, e_imsrg * phys_unit))

import matplotlib.pyplot as plt

s_vals = [x[0] for x in integration_data]
e_vals = [x[1] for x in integration_data]
plt.plot(s_vals, e_vals)
plt.xlim(0,10.0)
plt.show()
