import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / ".."))
import NuLattice.one_body_operators as obops
import NuLattice.two_body_operators as tbops
import NuLattice.lattice as lat
if __name__ == '__main__':
    thisL = 4
    a = 1.0 / 100.0
    lattice = lat.get_lattice(thisL)

    myTkin=obops.tKin(thisL, 3, a)
    print("number of matrix elements from kinetic energy", len(myTkin))

    bpi = 0.7
    verbose = False
    v_OPE = tbops.onePionEx(thisL, bpi, a, lattice, verbose=verbose)
    sNL = 0.08
    sL = 0.08
    c0 = -0.185

    v_0=tbops.shortRangeV_2body(lattice, thisL, sL, sNL, c0, a, verbose=verbose)

    mycontact = tbops.sparse_to_list_2body(v_0+v_OPE, thisL)
    print("number of matrix elements from two-body contacts", len(mycontact))
