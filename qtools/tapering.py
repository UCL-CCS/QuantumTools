from openfermion.ops import QubitOperator
import numpy as np
import scipy as sp
import sympy
from copy import deepcopy
import random
from openfermion.linalg import get_sparse_operator
from openfermion.utils import count_qubits
from functools import reduce
from typing import Tuple
import itertools

def pauliword_to_symplectic(n_qubits: int, QubitOp: QubitOperator) -> np.array:
    """
    Function that converts Openfermion QubitOperator to sympletic Pauli vector

    Args:
        pauliword (QubitOperator): Openfermion QubitOperator
        n_qubits (int): number of qubits (defines size of sympletic vector).

    Returns:
        sym_pvec (np.array): sympletic vector of Pauli operator (x_{0} ... x_{n-1} || z_{0} ... z_{n-1} )
    """

    sym_pvec = np.zeros((2 * n_qubits), dtype=int)

    qNo_Pstr_tuple = list(QubitOp.terms.keys())[0]
    for qNo, Pstr in qNo_Pstr_tuple:
        if Pstr == 'X':
            sym_pvec[qNo] = 1

        elif Pstr == 'Z':
            sym_pvec[qNo+n_qubits] = 1

        elif Pstr == 'Y':
            sym_pvec[qNo] = 1
            sym_pvec[qNo+n_qubits] = 1
        else:
            raise ValueError(f'not Pauli operator: {Pstr} on qubit {qNo}')

    return sym_pvec


def symplectic_to_pauliword(symplectic_array: np.array, coeff: float =1.0) -> QubitOperator:
    """
    Function that converts sympletic Pauli to Openfermion Qubit Operator

    Args:
        symplectic_array (np.array): sympletic vector of Pauli operator (x_{0} ... x_{n-1} || z_{0} ... z_{n-1} )
        coeff (float): optional coefficent for QubitOperator (defaults to +1)

    Returns:
        pauliword (QubitOperator): Openfermion QubitOperator
    """
    pauliword_str = ''
    n_qubits = len(symplectic_array) // 2
    for qubit_ind in range(n_qubits):
        if (symplectic_array[qubit_ind] + symplectic_array[qubit_ind+n_qubits]) == 2:
            pauliword_str+= f'Y{qubit_ind} '
        elif symplectic_array[qubit_ind]==1:
            pauliword_str += f'X{qubit_ind} '
        elif symplectic_array[qubit_ind+n_qubits]==1:
            pauliword_str += f'Z{qubit_ind} '

    pauliword = QubitOperator(pauliword_str, coeff)
    return pauliword


def build_G_matrix(list_qubit_operators: list, n_qubits: int) -> np.array:
    """
    Function to build G matrix according to page 11 of https://arxiv.org/pdf/1701.08213.pdf)

    G = [   Gx  ]
        [   --  ]
        [   Gz  ]

    In standard each column of G represents a particular Pauli operator of a Pauli Hamiltonian.

    Args:
        list_qubit_operators (list): List of openfermion QubitOperators to create G matrix from
        n_qubits (int): number of qubits (needed to define sympletic vector)

    Returns:
        G_matrix (np.array): G matrix of input list of qubit operators
    """

    G_mat = np.zeros((2*n_qubits, len(list_qubit_operators)), dtype=int)

    for col_ind, qubit_op in enumerate(list_qubit_operators):
        p_symp = pauliword_to_symplectic(n_qubits, qubit_op)
        G_mat[:, col_ind] = p_symp

    return G_mat


def build_E_matrix(G_matrix: np.array) -> np.array:
    """
    Function to build E matrix according to equation 60  of https://arxiv.org/pdf/1701.08213.pdf (pg11)

    E = [   Gz ^{T}  |   Gx^{T}  ]

    Importantly the kernal of E ( any vector v that results in E.dot(v) = 0   )
    represents terms that must commute!

    aka it computes all the terms where:


    Args:
        G_matrix (np.array): G matrix - columns represent Pauli operators in sympletic form
    Returns:
        E_matrix (np.array): E matrix
    """
    n_qubits = G_matrix.shape[1]//2
    Gx = G_matrix[:n_qubits, :]
    Gz = G_matrix[n_qubits:, :]

    E_matrix = np.block([Gz.T, Gx.T])

    return E_matrix


def gf2elim(M: np.array) -> np.array:
    """
    Function that performs Gaussian elimination over GF2(2)
    GF is the initialism of Galois field, another name for finite fields.

    GF(2) may be identified with the two possible values of a bit and to the boolean values true and false.

    Args:
        M (np.array): GF(2) binary matrix to preform Gaussian elimination over
    Returns:
        M (np.array): reduced row echelon form of M
    """
    m_rows, n_cols = M.shape

    row_i = 0
    col_j = 0

    while row_i < m_rows and col_j < n_cols:

        # find value and index of largest element in remainder of column j
        k = np.argmax(M[row_i:, col_j]) + row_i
        # + row_i gives correct index for largest value in column j (given we have started from row_i!)

        # swap rows
        temp = np.copy(M[k])
        M[k] = M[row_i]
        M[row_i] = temp

        aijn = M[row_i, col_j:]

        col = np.copy(M[:, col_j])  # make a copy otherwise M will be directly affected

        # zero out i th row (to avoid xoring pivot row with itself)
        col[row_i] = 0

        flip = np.outer(col, aijn)

        M[:, col_j:] = M[:, col_j:] ^ flip

        row_i += 1
        col_j += 1

    return M


def get_kernel(mat: np.array) -> np.array:
    """
    Function to get the kernal of a matrix (uses scipy)

    Args:
        mat (np.array): Matrix to get kernel of.
    Returns:
        null_space_vecs (np.array): the kernel of mat. Note that the columns of this array define the null vectors!
    """
    # columns define each vector (where mat @ vec = zero_vec)

    null_space_vecs = sp.linalg.null_space(mat, rcond=None)

    return null_space_vecs


def find_span_of_vectors(matrix_of_vectors) -> Tuple[np.array, np.array]:
    """
    Function to get span of a given set of vectors. This uses sympy and computes the reduced row echelon form
    of the input.

    Args:
        matrix_of_vectors (np.array): Matrix of vectors to give
    Returns:
        linearly_ind_vecs_array (np.array): array of linearly independent vectors
        pivots_array (np.array): array of indices where pivot locations are
    """
    m = sympy.Matrix(matrix_of_vectors)
    # Compute reduced row echelon form (rref).
    m_rref, pivots = m.rref()

    # convert to numpy
    linearly_ind_vecs_array = np.abs(np.array(m_rref, dtype=int))
    pivots_array = np.abs(np.array(pivots, dtype=int))
    return linearly_ind_vecs_array, pivots_array


def apply_pi4_rotation(rot_pauilword_A, Pauliword_B):
    """
    apply: R Pb R^dagger

           = [cos(pi/2) I + 1j sin(pi/2) A ] B [cos(pi/2) I - 1j sin(pi/2) A ] = [d I + 1j k A ] B [d - 1j k A ]

            = [d B + 1jk AB] [d - 1j k A ] = d^{2} B -1j d k BA + 1j d k AB + k^{2} ABA

            if [A,B] = 0 >> AB == BA
                then: d^{2} B -1j d k BA + 1j d k AB + k^{2} ABA
                    = d^{2} B + k^{2} B = B

            else {A,B} = 0 >> AB == -BA
                then: d^{2} B -1j d k BA + 1j d k AB + k^{2} ABA
                    = d^{2} B -2j d k BA - k^{2} B
                    = -2j d k BA
                    # as d^{2} - k^{2} = 0 FOR CASE WHEN PI/4 is used!

    """
    BA = Pauliword_B * rot_pauilword_A
    if rot_pauilword_A*Pauliword_B == BA:
        return rot_pauilword_A * Pauliword_B * rot_pauilword_A
    else:
        d = np.cos(np.pi/4)
        k = np.sin(np.pi/4)
        return -2j * d * k * BA


def Clifford_operator_reduction_to_X(list_symmetry_generators):

    single_qubit_generators = []
    rotations = []

    seen_qubits = set()
    for sym_ind in range(len(list_symmetry_generators)):
        symmetry_generator = list_symmetry_generators[sym_ind]

        paulis, coeff = zip(*symmetry_generator.terms.items())
        qNos, sigmas = zip(*paulis[0])

        if len(sigmas)==1 and ''.join(sigmas) == 'X':
            # already single qubit Pauli X!
            single_qubit_generators.append(symmetry_generator)
            continue
        else:
            if ''.join(sigmas) == 'X'*len(qNos):
                # all X case
                # select unique X term
                X_indices = set(q_ind for q_ind, sigma in zip(qNos, sigmas) if sigma == 'X')
                unqiue_inds = list(X_indices.difference(seen_qubits))
                qNo_selected = unqiue_inds[0]

                # map to single  Z term
                P_rot_diag = QubitOperator(f'Y{qNo_selected}', 1)

                # analytical
                non_diagonal_op = apply_pi4_rotation(P_rot_diag, symmetry_generator)
                rotations.append(P_rot_diag)

                # apply to generators
                list_symmetry_generators = [apply_pi4_rotation(P_rot_diag, op) for op in list_symmetry_generators]
            else:
                non_diagonal_op = symmetry_generator

            # map to non_diagonal_op term single qubit X
            paulis_nondiag, coeff_nondiag = zip(*non_diagonal_op.terms.items())
            qNos_nondiag, sigmas_nondiag = zip(*paulis_nondiag[0])

            if 'Z' in sigmas_nondiag:
                Z_indices = set(q_ind for q_ind, sigma in zip(qNos_nondiag, sigmas_nondiag) if sigma == 'Z')
                unqiue_inds = list(Z_indices.difference(seen_qubits))
                qNo_nondiag_selected = unqiue_inds[0]
                pauli_needed = 'Y'
            elif 'Y' in sigmas_nondiag:
                Y_indices = set(q_ind for q_ind, sigma in zip(qNos_nondiag, sigmas_nondiag) if sigma == 'Y')
                unqiue_inds = list(Y_indices.difference(seen_qubits))
                qNo_nondiag_selected = unqiue_inds[0]
                pauli_needed = 'Z'
            else:
                raise ValueError(f'{non_diagonal_op} is not a nondiagonal Pauliword')

            P_rot_nondiag = QubitOperator(' '.join([f'{pauli_needed}{qNo_nondiag_selected}' if qNo==qNo_nondiag_selected else f'{s}{qNo}'
                                 for qNo, s in zip(qNos_nondiag, sigmas_nondiag)]), 1)

            # analytical
            reduced_op = apply_pi4_rotation(P_rot_nondiag, non_diagonal_op)
            rotations.append(P_rot_nondiag)

            seen_qubits.add(list(*reduced_op.terms.keys())[0][0])
            single_qubit_generators.append(reduced_op)

            # apply to generators
            list_symmetry_generators = [apply_pi4_rotation(P_rot_nondiag, op) for op in list_symmetry_generators]

    return single_qubit_generators, rotations


def get_rotated_operator_and_generators(pauli_hamiltonian: QubitOperator) -> QubitOperator:

    n_qubits = count_qubits(pauli_hamiltonian)

    G_mat= build_G_matrix(list(pauli_hamiltonian), n_qubits)
    E_mat = build_E_matrix(G_mat)

    E_tilda = gf2elim(E_mat)

    # remove rows of zeros!
    E_tilda = E_tilda[~np.all(E_tilda == 0, axis=1)]

    null_vecs = get_kernel(E_tilda)

    linearly_ind_vecs_array, pivots_array = find_span_of_vectors(null_vecs.T)

    symmetry_generators = []
    for stabilizer in linearly_ind_vecs_array:
        Pword = symplectic_to_pauliword(stabilizer)
        symmetry_generators.append(Pword)

    generators, clifford_rotations = Clifford_operator_reduction_to_X(symmetry_generators)

    H_tapered = deepcopy(pauli_hamiltonian)
    for rot in clifford_rotations[::-1]:
        H_new = QubitOperator()
        for op in H_tapered:
            H_new += apply_pi4_rotation(rot, op)
        H_tapered = H_new

    return H_tapered, generators, clifford_rotations


def find_sector_brute_force(rotated_hamiltonian: QubitOperator, symmetry_generators: list) -> QubitOperator:

    for measurement_assignment in itertools.product([+1, -1], repeat=len(symmetry_generators)):
        pass
    pass


def give_sector_from_input_state():
    pass

if __name__ ==  '__main__':

    H = [QubitOperator('Z0',           random.uniform(5, 10)),
         QubitOperator('Z1',           random.uniform(5, 10)),
         QubitOperator('Z2',           random.uniform(5, 10)),
         QubitOperator('Z3',           random.uniform(5, 10)),
         QubitOperator('Z0 Z1',        random.uniform(5, 10)),
         QubitOperator('Z0 Z2',        random.uniform(5, 10)),
         QubitOperator('Z0 Z3',        random.uniform(5, 10)),
         QubitOperator('Z1 Z2',        random.uniform(5, 10)),
         QubitOperator('Z1 Z3',        random.uniform(5, 10)),
         QubitOperator('Z2 Z3',        random.uniform(5, 10)),
         QubitOperator('Y0 Y1 X2 X3 ', random.uniform(5, 10)),
         QubitOperator('X0 Y1 Y2 X3 ', random.uniform(5, 10)),
         QubitOperator('Y0 X1 X2 Y3 ', random.uniform(5, 10)),
         QubitOperator('X0 X1 Y2 Y3 ', random.uniform(5, 10)),
         ]

    G = build_G_matrix(H, 4)
    E = build_E_matrix(G)
    E_tilda = gf2elim(E)
    E_tilda = E_tilda[~np.all(E_tilda == 0, axis=1)] # remove rows of zeros!

    null_vecs = get_kernel(E_tilda)

    linearly_ind_vecs_array, pivots_array = find_span_of_vectors(null_vecs.T)
    print(linearly_ind_vecs_array)
    symmetry_generators = []
    for stabilizer in linearly_ind_vecs_array:
        Pword = symplectic_to_pauliword(stabilizer)
        print(Pword)
        symmetry_generators.append(Pword)

    generators, rotations = Clifford_operator_reduction_to_X(symmetry_generators)

    H_tapered = deepcopy(H)

    for rot in rotations[::-1]:
        H_new = QubitOperator()
        for op in H_tapered:
            H_new += apply_pi4_rotation(rot, op)
        H_tapered = list(H_new)


    H_tapered = reduce(lambda x, y: x + y, H)
    for rot in rotations[::-1]:
        rot_op = QubitOperator('', np.cos(np.pi / 4)) + 1j * np.sin(np.pi / 4) * rot
        rot_op_dag = QubitOperator('', np.cos(np.pi / 4)) - 1j * np.sin(np.pi / 4) * rot
        H_tapered = rot_op * H_tapered * rot_op_dag

    H_old_mat = get_sparse_operator(reduce(lambda x,y:x+y, H))
    H_new_mat = get_sparse_operator(reduce(lambda x, y: x + y, H_tapered))

    from numpy.linalg import eigh
    eigvecs, eigvals = eigh(H_new_mat.todense())
    eigvecs2, eigvals2 = eigh(H_old_mat.todense())

    print(eigvecs)
    print(eigvecs2)
    print(H_tapered)



