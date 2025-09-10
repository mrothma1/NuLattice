# `NuLattice`: Ab initio computations of atomic nuclei on lattices

`NuLattice` is a set of Python programs (full configuration interaction,
coupled cluster method, Hartree Fock, in-medium similarity
renormalization group) for computations of atomic nuclei on spatial
lattices with periodic boundary conditions. At present, the available
Hamiltonians are based on pionless effective field theory and consist
of the kinetic energy, an on-site (SU4 symmetric) two-body contact,
and an on-site three-body contact.

[![Documentation Status](https://app.readthedocs.org/projects/nulattice/badge/)](https://nulattice.readthedocs.io/)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.17094173.svg)](https://doi.org/10.5281/zenodo.17094173)

## Setup

To set up the repository to run the code, install the required packages in `requirements.txt`.

We recommend setting up a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
This virtual environment can always be (re)activated with:
```
source .venv/bin/activate
```

Once the virtual environment has been set up, 
the scripts in `examples` and `benchmarks` can simply be run as (for example):
```
python3 Example_CCSD.py
```

## Contributing

To contribute to `NuLattice,` please fork the repository, 
create a new branch with your feature, 
and once it is complete open a pull request to contribute to the main project.
Consistent contributors will be given access to the organization
to easily contribute to the project.

