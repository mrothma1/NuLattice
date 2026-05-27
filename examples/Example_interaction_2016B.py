import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".."))
import NuLattice.operators.one_body_operators as obo
import NuLattice.operators.two_body_operators as twbo
import NuLattice.lattice as lat
import NuLattice.constants_NLEFT as nleftConsts
import NuLattice.references as ref
import NuLattice.HF.hartree_fock as hf
import numpy as np
import NuLattice.FCI.few_body_diagonalization as fbd
from scipy.sparse.linalg import eigsh as arpack_eigsh
if __name__ == '__main__':
    myL = 3
    a = 1.0 / 100.0
    lattice = lat.get_lattice(myL)

    myTkin=obo.tKin(myL, 3, a, mass = nleftConsts.mass)
    print("number of matrix elements from kinetic energy", len(myTkin))

    bpi = 0.7
    verbose = True
    cNL = -0.1171 / a
    sNL = 0.077
    cINL = 0.02607 / a
    v_OPE = twbo.onePionEx(myL, bpi, a, lattice, verbose=verbose)
    v_NL=twbo.shortRangeV_2body(lattice, myL, 0, sNL, cNL , verbose=verbose)
    iso_ops = [obo.tau_x(lattice, myL), obo.tau_y(lattice, myL), obo.tau_z(lattice, myL)]
    for op in iso_ops:
        v_NL += twbo.shortRangeV_2body(lattice, myL, 0, sNL, cINL, verbose = verbose, op1b = obo.list_to_sparse1b(op) * 2)

    cL = -0.01013 / a
    sL = 0.81
    cSL = - cL / 3.0
    cIL = cSL
    cSIL = cSL
    sp_ops = [obo.spin_x(lattice, myL), obo.spin_y(lattice, myL), obo.spin_z(lattice, myL)]
    v_L = twbo.shortRangeV_2body(lattice, myL, sL, 0, cL, verbose=verbose)
    for op in sp_ops:
        v_L += twbo.shortRangeV_2body(lattice, myL, sL, 0, cSL, verbose = verbose, op1b = obo.list_to_sparse1b(op) * 2)
    for op in iso_ops:
        v_L += twbo.shortRangeV_2body(lattice, myL, sL, 0, cIL, verbose = verbose, op1b = obo.list_to_sparse1b(op) * 2)
        for op2 in sp_ops:
            op1b = obo.list_to_sparse1b(op) @ obo.list_to_sparse1b(op2) 
            v_L += twbo.shortRangeV_2body(lattice, myL, sL, 0, cSL, verbose = verbose, op1b = op1b * 4)
    
    mycontact = twbo.sparse_to_list_2body(v_NL+v_L+v_OPE, myL)
    print("number of matrix elements from two-body contacts", len(mycontact))
