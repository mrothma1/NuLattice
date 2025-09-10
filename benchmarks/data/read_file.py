# Copyright 2025 Matthias Heinz. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Module to load local benchmark data."""
__authors__ = ["Matthias Heinz"]
__credits__ = ["Matthias Heinz"]
__copyright__ = "(c) Matthias Heinz"
__license__ = "BSD-3-Clause"
__date__ = "2025-09-03"

import numpy as np
import pathlib


def expand_path_to_file(file_path):
    """
    Expands a relative file path to the absolute path in the data directory

    Extracts the filename from a path and constructs the full path relative
    to the location of this module's data directory

    :param file_path:   Input file path (can be relative or just filename)
    :type file_path:    str
    :return:            Absolute path to the file in the data directory
    :rtype:             str
    """
    fname = file_path.split("/")[-1]
    data_dir = pathlib.Path(__file__).parent

    return str(data_dir / fname)


def read_file_basis_occs(file):
    """
    Reads occupation numbers from the corresponding basis file

    Extracts occupation numbers for each single-particle state from the
    basis file, which contains state information and occupation values

    :param file:    Path to operator file (automatically converted to basis file path)
    :type file:     str
    :return:        Array of occupation numbers for each basis state
    :rtype:         numpy array
    """
    file = file.replace("0B.dat", "basis.dat")
    file = file.replace("1B.dat", "basis.dat")
    file = file.replace("2B.dat", "basis.dat")
    occs = []
    file = expand_path_to_file(file)
    with open(file) as fin:
        for line in fin:
            occs.append(float(line.strip().split()[-1]))
    return np.array(occs)


def read_file_basis_size(file):
    """
    Determines the dimension of the single-particle basis from file

    Counts the number of single-particle states by reading the basis file
    and returning the length of the occupation array

    :param file:    Path to operator file (used to locate corresponding basis file)
    :type file:     str
    :return:        Dimension of the single-particle basis
    :rtype:         int
    """
    return len(read_file_basis_occs(file))


def read_file_0b(file):
    """
    Reads a zero-body (scalar) operator value from file

    Loads a single floating-point value representing a scalar quantity
    such as the zero-body energy contribution

    :param file:    Path to the input file containing scalar value
    :type file:     str
    :return:        Zero-body operator value
    :rtype:         float
    """
    with open(file) as fin:
        return float(fin.readline())


def read_file_1b(file):
    """
    Reads one-body operator matrix elements from file

    Loads one-body operator matrix elements from a file containing
    lines in the format: index_p index_q matrix_element

    :param file:    Path to the one-body operator file
    :type file:     str
    :return:        One-body operator matrix
    :rtype:         numpy array
    """
    dim = read_file_basis_size(file)
    op = np.zeros((dim, dim))
    with open(file) as fin:
        for line in fin:
            p, q = [int(x) for x in line.strip().split()[:-1]]
            me = float(line.strip().split()[-1])
            op[p, q] = me
    return op


def read_file_2b(file):
    """
    Reads two-body operator matrix elements from file

    Loads two-body operator matrix elements from a file containing
    lines in the format: index_p index_q index_r index_s matrix_element

    :param file:    Path to the two-body operator file
    :type file:     str
    :return:        Two-body operator tensor
    :rtype:         numpy array
    """
    dim = read_file_basis_size(file)
    op = np.zeros((dim, dim, dim, dim))
    with open(file) as fin:
        for line in fin:
            p, q, r, s = [int(x) for x in line.strip().split()[:-1]]
            me = float(line.strip().split()[-1])
            op[p, q, r, s] = me
    return op


def read_file(file):
    """
    Reads operator data from file based on filename convention

    Automatically determines the operator type (0-body, 1-body, or 2-body)
    from the filename suffix and calls the appropriate reading function

    :param file:    Path to the operator file with suffix indicating type
    :type file:     str
    :return:        Operator data (scalar, matrix, or tensor depending on type)
    :rtype:         float | numpy array
    :raises Exception: If filename does not match expected conventions
    """
    file = expand_path_to_file(file)
    if "_0B.dat" in file:
        return read_file_0b(file)
    if "_1B.dat" in file:
        return read_file_1b(file)
    if "_2B.dat" in file:
        return read_file_2b(file)
    raise Exception("Do not know how to handle file: {}".format(file))
