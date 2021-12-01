from openfermion.ops import QubitOperator
import numpy as np
import scipy as sp
import sympy
from copy import deepcopy
import random
from openfermion.linalg import get_sparse_operator
from openfermion.utils import count_qubits
from functools import reduce
from typing import Tuple, List
import itertools
from scipy.sparse.linalg import eigsh


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
    n_qubits = G_matrix.shape[0]//2
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


def gf2elim(M):

    m,n = M.shape

    i=0
    j=0

    while i < m and j < n:
        # find value and index of largest element in remainder of column j
        k = np.argmax(M[i:, j]) +i

        # swap rows
        #M[[k, i]] = M[[i, k]] this doesn't work with numba
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp


        aijn = M[i, j:]

        col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected

        col[i] = 0 #avoid xoring pivot row with itself

        flip = np.outer(col, aijn)

        M[:, j:] = M[:, j:] ^ flip

        i += 1
        j +=1

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


def apply_pi4_rotation(rot_pauilword_A: QubitOperator, Pauliword_B: QubitOperator) -> QubitOperator:
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

    Args:
        rot_pauilword_A (QubitOperator): qubit operator to rotate with
        Pauliword_B (QubitOperator): qubit operator rotate
    Returns:
        The conjugate rotated Pauliword_B operator

    """
    BA = Pauliword_B * rot_pauilword_A
    if rot_pauilword_A*Pauliword_B == BA:
        return rot_pauilword_A * Pauliword_B * rot_pauilword_A
    else:
        d = np.cos(np.pi/4)
        k = np.sin(np.pi/4)
        return -2j * d * k * BA


def Clifford_operator_reduction_to_X(list_symmetry_generators) -> Tuple[list, list]:
    """
    Function maps list of symmetry operators to single qubit Pauli X operators using Clifford rotations

    Args:
        list_symmetry_generators (list): list of INDEPENDENT symmetry operators (QubitOperator)
    Returns:
        single_qubit_generators (list): list of symmetry operators that have been mapped to single Pauli X terms
        rotations (list): list of cliford rotations used to reduce all symmetry operators to single Pauli X terms
    """
    single_qubit_generators = []
    rotations = []
    list_symmetry_generators = deepcopy(list_symmetry_generators)

    seen_qubits = set()
    while list_symmetry_generators:
        symmetry_generator = list_symmetry_generators.pop(0)

        sym_gen_sig_qno, coeff = zip(*symmetry_generator.terms.items())
        sym_qno, sym_paulis = zip(*sym_gen_sig_qno[0])

        if len(sym_paulis)==1 and ''.join(sym_paulis) == 'X':
            # already single qubit Pauli X!
            single_qubit_generators.append(symmetry_generator)
            continue
        else:
            # symmetry_generator requires mapping to single qubit Pauli X
            if ''.join(sym_paulis) == 'X'*len(sym_qno):
                # all X case

                # find unique qubit indices to turn into Pauli Z (note currently all Pauli Xs)
                X_indices = set(sym_qno)
                unqiue_inds = list(X_indices.difference(seen_qubits))
                qNo_for_X_to_Z = unqiue_inds[0]

                # map unique qubit X to single  Z term
                pauli_Y_rot = QubitOperator(f'Y{qNo_for_X_to_Z}', 1)

                # apply rotation to all qubit X operator to generate a single Z operator on qNo_for_X_to_Z
                op_with_one_Z_or_Y = apply_pi4_rotation(pauli_Y_rot, symmetry_generator)
                rotations.append(op_with_one_Z_or_Y)

                # apply to generators
                list_symmetry_generators = [apply_pi4_rotation(pauli_Y_rot, op) for op in list_symmetry_generators]
            else:
                op_with_one_Z_or_Y = symmetry_generator

            # map operator containing at least one Z and Y on a single qubit to single Pauli X operator!
            Z_Y_single_qop_sig_qno, Z_Y_single_coeff = zip(*op_with_one_Z_or_Y.terms.items())
            Z_Y_single_qNos, Z_Y_single_paulis = zip(*Z_Y_single_qop_sig_qno[0])

            if 'Z' in Z_Y_single_paulis:
                Z_indices = set(q_ind for q_ind, sigma in zip(Z_Y_single_qNos, Z_Y_single_paulis) if sigma == 'Z')
                unqiue_inds = list(Z_indices.difference(seen_qubits))
                qNo_to_map_to_X = unqiue_inds[0]
                pauli_needed = 'Y'
            elif 'Y' in Z_Y_single_paulis:
                Y_indices = set(q_ind for q_ind, sigma in zip(Z_Y_single_qNos, Z_Y_single_paulis) if sigma == 'Y')
                unqiue_inds = list(Y_indices.difference(seen_qubits))
                qNo_to_map_to_X = unqiue_inds[0]
                pauli_needed = 'Z'
            else:
                raise ValueError(f'{op_with_one_Z_or_Y} does not contain single qubit Z or Y')

            # build operator to map op_with_one_Z_or_Y to single qubit Z (only differs on
            P_rot_to_single_X = QubitOperator(' '.join([f'{pauli_needed}{qNo_to_map_to_X}' if qNo==qNo_to_map_to_X else f'{s}{qNo}'
                                 for qNo, s in zip(Z_Y_single_qNos, Z_Y_single_paulis)]), Z_Y_single_coeff[0])

            # analytical
            reduced_op = apply_pi4_rotation(P_rot_to_single_X, op_with_one_Z_or_Y)
            rotations.append(P_rot_to_single_X)

            seen_qubits.add(list(*reduced_op.terms.keys())[0][0])
            single_qubit_generators.append(reduced_op)

            # apply to generators
            list_symmetry_generators = [apply_pi4_rotation(P_rot_to_single_X, op) for op in list_symmetry_generators]

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

    single_X_generators, clifford_rotations = Clifford_operator_reduction_to_X(symmetry_generators)

    H_rotated = deepcopy(pauli_hamiltonian)
    for rot in clifford_rotations[::-1]:
        H_new = QubitOperator()
        for op in H_rotated:
            H_new += apply_pi4_rotation(rot, op)
        H_rotated = H_new

    return H_rotated, symmetry_generators, single_X_generators, clifford_rotations


def find_sector_brute_force(rotated_hamiltonian: QubitOperator, symmetry_generators: list) -> List:

    symmetry_generator_qubit_indices = set()
    for symm_op in symmetry_generators:
        for Pauliword, coeff in symm_op.terms.items():
            qNos, _ = zip(*Pauliword)
            symmetry_generator_qubit_indices.add(*qNos)

    # symmetry_generator_signs = [coeff for term in symmetry_generators for coeff in term.terms.values()]

    # build dictionary to map qubits to new indices
    n_qubits = count_qubits(rotated_hamiltonian)
    qubit_inds_kept = set(range(n_qubits)).difference(symmetry_generator_qubit_indices)
    qubit_relabel = dict(zip(qubit_inds_kept, range(len(qubit_inds_kept))))
    all_possible_H: List[QubitOperator] = []
    # iterate over all possible 2^n_generator +1/-1 measurement assignments
    for measurement_assignment in itertools.product([+1, -1], repeat=len(symmetry_generators)):

        # generator_ind_and_meas_val = dict(zip(symmetry_generator_qubit_indices,
        #                                       list(map(lambda x,y: x*y, symmetry_generator_signs,
        #                                                                 measurement_assignment))))
        generator_ind_and_meas_val = dict(zip(symmetry_generator_qubit_indices,measurement_assignment))

        H_reduced = QubitOperator()
        for h_op in rotated_hamiltonian:
            for Pauliword, coeff in h_op.terms.items():
                new_coeff = coeff
                qNos, paulis = zip(*Pauliword)
                for qNo in set(qNos).intersection(symmetry_generator_qubit_indices):
                    new_coeff = new_coeff*generator_ind_and_meas_val[qNo]

            qNos_reduced = set(qNos).difference(symmetry_generator_qubit_indices)
            Pauliword_reduced = ' '.join([f'{Pauli}{qubit_relabel[qNo]}'for qNo, Pauli in zip(qNos, paulis)
                                          if qNo in qNos_reduced])

            H_reduced += QubitOperator(Pauliword_reduced, new_coeff)
        all_possible_H.append(H_reduced)

    # find eig vals of all possible terms
    H_red_eig_vals=[]
    for H_red in all_possible_H:
        sparse_H_red = get_sparse_operator(H_red)
        if sparse_H_red.shape[0]< 128:
            eigvals, eigvecs = np.linalg.eigh(sparse_H_red.todense())
            min_eigval = min(eigvals)
        else:
            min_eigval = eigsh(sparse_H_red, k = 1, which = 'SA')[0][0]

        H_red_eig_vals.append(min_eigval)

    return sorted(list(zip(H_red_eig_vals, all_possible_H)), key=lambda x:x[0])


def find_ground_sector_using_input_state(rotated_hamiltonian: QubitOperator, symmetry_generators_pre_rotatation: list,
                                       qubit_state: np.array, symmetry_generators: list,
                                       check_correct: bool = False) -> QubitOperator:




    qubit_state_Z_mesaure = -1*((2*qubit_state) - 1) # plus and minus one for Z measurement
    generator_ind_and_meas_val = {}
    for sym_op_ind, symm_op in enumerate(symmetry_generators_pre_rotatation):

        sym_gen_sig_qno, coeff = zip(*symm_op.terms.items())
        sym_qno, sym_paulis = zip(*sym_gen_sig_qno[0])
        if not ''.join(sym_paulis) == 'Z' * len(sym_qno):
            raise ValueError(f'Currently function only works for Z generators! {symm_op} not valild')
        generator_ind_and_meas_val[sym_op_ind] = np.prod(np.take(qubit_state_Z_mesaure, sym_qno)) * coeff[0]


    symmetry_generator_qubit_indices = set()
    for symm_op in symmetry_generators:
        for Pauliword, coeff in symm_op.terms.items():
            qNos, _ = zip(*Pauliword)
            symmetry_generator_qubit_indices.add(*qNos)

    symmetry_generator_signs = [coeff for term in symmetry_generators for coeff in term.terms.values()]
    for ind, sign in enumerate(symmetry_generator_signs):
        generator_ind_and_meas_val[ind]*=sign

    # build dictionary to map qubits to new indices
    n_qubits = count_qubits(rotated_hamiltonian)
    qubit_inds_kept = set(range(n_qubits)).difference(symmetry_generator_qubit_indices)
    qubit_relabel = dict(zip(qubit_inds_kept, range(len(qubit_inds_kept))))


    H_reduced = QubitOperator()
    for h_op in rotated_hamiltonian:
        for Pauliword, coeff in h_op.terms.items():
            new_coeff = coeff
            qNos, paulis = zip(*Pauliword)
            for qNo in set(qNos).intersection(symmetry_generator_qubit_indices):
                new_coeff = new_coeff*generator_ind_and_meas_val[qNo]

        qNos_reduced = set(qNos).difference(symmetry_generator_qubit_indices)
        Pauliword_reduced = ' '.join([f'{Pauli}{qubit_relabel[qNo]}'for qNo, Pauli in zip(qNos, paulis)
                                      if qNo in qNos_reduced])

        H_reduced += QubitOperator(Pauliword_reduced, new_coeff)

    if check_correct:
        sparse_H_full = get_sparse_operator(rotated_hamiltonian)
        if sparse_H_full.shape[0] < 128:
            eigvals, eigvecs = np.linalg.eigh(sparse_H_full.todense())
            min_eigval_full = min(eigvals)
        else:
            min_eigval_full = eigsh(sparse_H_full, k=1, which='SA')[0][0]

        sparse_H_red = get_sparse_operator(H_reduced)
        if sparse_H_red.shape[0]< 128:
            eigvals, eigvecs = np.linalg.eigh(sparse_H_red.todense())
            min_eigval = min(eigvals)
        else:
            min_eigval = eigsh(sparse_H_red, k = 1, which = 'SA')[0][0]

        if not np.isclose(min_eigval_full, min_eigval):
            print(min_eigval_full)
            print(min_eigval)
            raise ValueError('wrong sector found')

    return H_reduced


if __name__ ==  '__main__':

    JW_ket = np.eye(2**4)[int('0101',2)].reshape(16,1)
    JW_array = np.array([0,1,0,1])

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

    H_rotated = deepcopy(H)

    for rot in rotations:
        H_new = QubitOperator()
        for op in H_rotated:
            H_new += apply_pi4_rotation(rot, op)
        H_rotated = H_new

    H_old_mat = get_sparse_operator(reduce(lambda x,y:x+y, H))
    H_new_mat = get_sparse_operator(reduce(lambda x, y: x + y, H_rotated))

    from numpy.linalg import eigh
    eigvals, eigvecs = eigh(H_new_mat.todense())
    eigvals2, eigvecs2 = eigh(H_old_mat.todense())

    print(np.allclose(eigvals, eigvals2))
    print(min(eigvals2))


    all_tapered_H = find_sector_brute_force(H_rotated, generators)
    print(all_tapered_H)

    H_tapered_ground = find_ground_sector_using_input_state(H_rotated, symmetry_generators,
                                                            JW_array, generators,check_correct = True)

    H_tapered_ground_mat = get_sparse_operator(reduce(lambda x,y:x+y, H_tapered_ground))
    eigvals3, eigvecs3 = eigh(H_tapered_ground_mat.todense())
    print()
    print('ground H tapered:')
    print(min(eigvals3))
    print(H_tapered_ground)