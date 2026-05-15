import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".."))
import NuLattice.operators.one_body_operators as obo
import NuLattice.operators.two_body_operators as twbo
import NuLattice.lattice as lat
import NuLattice.constants_NLEFT as nleftConsts
if __name__ == '__main__':
    myL = 4
    a = 1.0 / 100.0
    lattice = lat.get_lattice(myL)

    myTkin=obo.tKin(myL, 3, a,mass=nleftConsts.mass)
    print("number of matrix elements from kinetic energy", len(myTkin))

    bpi = 0.7
    verbose = True
    v_OPE = twbo.onePionEx(myL, bpi, a, lattice, verbose=verbose, g_A=nleftConsts.g_A, f_pi = nleftConsts.f_pi, m_pi_0=nleftConsts.m_pi_0)
    cNL = -0.1171 / a
    sNL = 0.077
    cINL = 0.02607 / a
    sL = 0
    v_NL=twbo.shortRangeV_2body(lattice, myL, sL, sNL, cNL, verbose=verbose)
    iso_ops = [obo.tau_x(lattice, myL), obo.tau_y(lattice, myL), obo.tau_z(lattice, myL)]
    for op in iso_ops:
        v_NL += twbo.shortRangeV_2body(lattice, myL, sL, sNL, cINL, verbose = verbose, op1b = obo.list_to_sparse1b(op))

    cL = -0.01013 / a
    sL = 0.81
    sNL = 0
    cSL = - cL / 3.0
    cIL = cSL
    cSIL = cSL
    sp_ops = [obo.spin_x(lattice, myL), obo.spin_y(lattice, myL), obo.spin_z(lattice, myL)]
    v_L = twbo.shortRangeV_2body(lattice, myL, sL, sNL, cL, verbose=verbose)
    for op in sp_ops:
        v_L += twbo.shortRangeV_2body(lattice, myL, sL, sNL, cSL, verbose = verbose, op1b = obo.list_to_sparse1b(op))
    for op in iso_ops:
        v_L += twbo.shortRangeV_2body(lattice, myL, sL, sNL, cIL, verbose = verbose, op1b = obo.list_to_sparse1b(op))
        for op2 in sp_ops:
            op1b = obo.list_to_sparse1b(op) @ obo.list_to_sparse1b(op2) 
            v_L += twbo.shortRangeV_2body(lattice, myL, sL, sNL, cSL, verbose = verbose, op1b = op1b)
    
    mycontact = twbo.sparse_to_list_2body(v_NL+v_L+v_OPE, myL)
    print("number of matrix elements from two-body contacts", len(mycontact))
