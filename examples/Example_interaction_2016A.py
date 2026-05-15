import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".."))
import NuLattice.operators.one_body_operators as obops
import NuLattice.operators.two_body_operators as tbops
import NuLattice.lattice as lat
import NuLattice.constants_NLEFT as nleftConsts
if __name__ == '__main__':
    thisL = 4
    a = 1.0 / 100.0
    lattice = lat.get_lattice(thisL)

    myTkin=obops.tKin(thisL, 3, a,mass=nleftConsts.mass)
    print("number of matrix elements from kinetic energy", len(myTkin))

    bpi = 0.7
    verbose = True
    v_OPE = tbops.onePionEx(thisL, bpi, a, lattice, verbose=verbose, g_A=nleftConsts.g_A, f_pi = nleftConsts.f_pi, m_pi_0=nleftConsts.m_pi_0)
    cNL = -0.2268 / a
    sNL = 0.077
    cINL = 0.02184 / a
    sL = 0
    v_NL=tbops.shortRangeV_2body(lattice, thisL, sL, sNL, cNL, verbose=verbose)
    iso_ops = [obops.tau_x(lattice, thisL), obops.tau_y(lattice, thisL), obops.tau_z(lattice, thisL)]
    for op in iso_ops:
        v_NL += tbops.shortRangeV_2body(lattice, thisL, sL, sNL, cINL, verbose = verbose, op1b = obops.list_to_sparse1b(op))
    mycontact = tbops.sparse_to_list_2body(v_NL+v_OPE, thisL)
    print("number of matrix elements from two-body contacts", len(mycontact))
