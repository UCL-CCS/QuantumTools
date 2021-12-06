from openfermion.ops import QubitOperator
import numpy as np
from copy import deepcopy
import random
from openfermion.linalg import get_sparse_operator
from openfermion.utils import count_qubits
from functools import reduce
from typing import Tuple, List
import itertools
from scipy.sparse.linalg import eigsh
from collections import namedtuple
from numpy.linalg import eigh

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


def gf2_gaus_elim(gf2_matrix: np.array) -> np.array:
    """
    Function that performs Gaussian elimination over GF2(2)
    GF is the initialism of Galois field, another name for finite fields.

    GF(2) may be identified with the two possible values of a bit and to the boolean values true and false.

    pseudocode: http://dde.binghamton.edu/filler/mct/hw/1/assignment.pdf

    Args:
        gf2_matrix (np.array): GF(2) binary matrix to preform Gaussian elimination over
    Returns:
        gf2_matrix_rref (np.array): reduced row echelon form of M
    """
    gf2_matrix_rref = gf2_matrix.copy()
    m_rows, n_cols = gf2_matrix_rref.shape

    row_i = 0
    col_j = 0

    while row_i < m_rows and col_j < n_cols:

        if sum(gf2_matrix_rref[row_i:, col_j]) == 0:
            # case when col_j all zeros
            # No pivot in this column, pass to next column
            col_j += 1
            continue

        # find index of row with first "1" in the vector defined by column j (note previous if statement removes all zero column)
        k = np.argmax(gf2_matrix_rref[row_i:, col_j]) + row_i
        # + row_i gives correct index (as we start search from row_i!)

        # swap row k and row_i (row_i now has 1 at top of column j... aka: gf2_matrix_rref[row_i, col_j]==1)
        gf2_matrix_rref[[k, row_i]] = gf2_matrix_rref[[row_i, k]]
        # next need to zero out all other ones present in column j (apart from on the i_row!)
        # to do this use row_i and use modulo addition to zero other columns!

        # make a copy of j_th column of gf2_matrix_rref, this includes all rows (0 -> M)
        Om_j = np.copy(gf2_matrix_rref[:, col_j])

        # zero out the i^th position of vector Om_j (this is why copy needed... to stop it affecting gf2_matrix_rref)
        Om_j[row_i] = 0
        # note this was orginally 1 by definition...
        # This vector now defines the indices of the rows we need to zero out
        # by setting ith position to zero - it stops the next steps zeroing out the i^th row (which we need as our pivot)


        # next from row_i of rref matrix take all columns from j->n (j to last column)
        # this is vector of zero and ones from row_i of gf2_matrix_rref
        i_jn = gf2_matrix_rref[row_i, col_j:]
        # we use i_jn to zero out the rows in gf2_matrix_rref[:, col_j:] that have leading one (apart from row_i!)
        # which rows are these? They are defined by that Om_j vector!

        # the matrix to zero out these rows is simply defined by the outer product of Om_j and i_jn
        # this creates a matrix of rows of i_jn terms where Om_j=1 otherwise rows of zeros (where Om_j=0)
        Om_j_dependent_rows_flip = np.einsum('i,j->ij', Om_j, i_jn, optimize=True)
        # note flip matrix is contains all m rows ,but only j->n columns!

        # perfrom bitwise xor of flip matrix to zero out rows in col_j that that contain a leading '1' (apart from row i)
        gf2_matrix_rref[:, col_j:] = np.bitwise_xor(gf2_matrix_rref[:, col_j:], Om_j_dependent_rows_flip)

        row_i += 1
        col_j += 1

    return gf2_matrix_rref


def gf2_basis_for_gf2_rref(gf2_matrix_in_rreform: np.array) -> np.array:
    """
    Function that gets the kernel over GF2(2) of ow reduced  gf2 matrix!

    uses method in: https://en.wikipedia.org/wiki/Kernel_(linear_algebra)#Basis

    Args:
        gf2_matrix_in_rreform (np.array): GF(2) matrix in row reduced form
    Returns:
        basis (np.array): basis for gf2 input matrix that was in row reduced form
    """
    rows_to_columns = gf2_matrix_in_rreform.T
    eye = np.eye(gf2_matrix_in_rreform.shape[1], dtype=int)

    # do column reduced form as row reduced form
    rrf = gf2_gaus_elim(np.hstack((rows_to_columns, eye.T)))

    zero_rrf = np.where(~rrf[:, :gf2_matrix_in_rreform.shape[0]].any(axis=1))[0]
    basis = rrf[zero_rrf, gf2_matrix_in_rreform.shape[0]:]

    return basis


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


def Global_commuting_terms_reduction_to_single_Q_ops(array_of_symmetry_generators_sympletic) -> list:
    """
    Function takes in an matrix of sympletic vectors that represent the symmetry generator operators that globally commute
    with the full Hamiltonian. (These are the basis for the kernel of Etilda).

    Output from gf2_basis_for_gf2_rref function can be used!

    and maps these Pauli operators to to single qubit Pauli operators using U operators. Which pauli is defined by
    anti_comm_pauli dictionary in code. This can be changed (but key value Paulis must anticommute).

    These U operators are NOT GENERAL CLIFFORD OPERATORS! However, as the symmetry generators globally
    commute with the full Hamiltonian they end up doing the same clifford operation. We explicitly write this here:

    U = 1/2(^0.5)(G_{j} + σ^{j})

    # σ = {X,Y,Z}
    # σ is chosen such that it anti-commutes with G_{j} and commutes with all other G_{i=!j} in G

    The action is as follows:

    U_{i} G_{i} U_{i} = 1/2 ( G_{i} + σ^{i} ) G_{i} ( G_{i} + σ^{i} )
                      = 1/2  ( I +  σ^{i} G_{i} ) ( G_{i} + σ^{i} )
                      = 1/2  ( G_{i} + σ^{i} +  σ^{i} + σ^{i} G_{i}σ^{i} )
                      = 1/2  ( G_{i} + σ^{i} +  σ^{i} -  G_{i} )      #using  {G_{i}, σ^{i}} = 0
                      = σ^{i}

    note the effect of U_{i} on the other generators (where we have set condition that [G_{j}, σ^{i}] = 0
    note for U_{i} G_{j} U_{i} = 1/2 ( G_{i} + σ^{i} ) G_{j} ( G_{i} + σ^{i} )
                               = 1/2 ( G_{i}G_{j} + σ^{i}G_{j} )  ( G_{i} + σ^{i} )
                               = 1/2 ( G_{i}G_{j}G_{i} + G_{i}G_{j}σ^{i} + σ^{i}G_{j} G_{i} + σ^{i}G_{j}σ^{i}  )
                               = 1/2 ( G_{j} + G_{i}G_{j}σ^{i} + G_{j}σ^{i} G_{i} + G_{j}  )   # using [G_{j}, σ^{i}] = 0
                               = 1/2 ( G_{j} + [G_{j}, σ^{i}]G_{i}  + G_{j}  )
                               = G_{j}


    Note as all the G_i terms commute globally with H... The action of U_{i} H U_{i} doesn't result in more terms.
    However, a general 1/2(^0.5)(P + σ^{j}) would result in more terms in H (aka have structure from global commutation!)
    Hence why we say that each U is a clifford operation.

    Args:
        array_of_symmetry_generators_sympletic (np.array): sympletic array of INDEPENDENT fully commuting symmetry operators
        each row represents an operator.
    Returns:
        list_clifford_like_rot (list): list of clifford rotations used to reduce all symmetry operators to single Pauli X terms.
        List of named tuples - containing (G_{j} + σ^{j}) ... See above description. Note we do not include 1/(2)^0.5
        as we can apply these operations symbolically. See Apply_U_tapering_symmetry_rotations function.
    """
    anti_comm_pauli = {
        'X': 'Z',
        'Y': 'Z',
        'Z': 'X'}

    # find unique qubit indices
    n_qubits = array_of_symmetry_generators_sympletic.shape[1] // 2

    # get unique qubit indices of commuting pauli generators
    column_sum = np.einsum('ij->j', array_of_symmetry_generators_sympletic)
    unqiue_qubit_indices = set(np.where(column_sum[:n_qubits] + column_sum[n_qubits:] <= 2)[0])

    U_op = namedtuple('U_op', 'Gi Si')
    list_clifford_like_rot = []
    for symmetry_op_sympletic in array_of_symmetry_generators_sympletic:

        symmetry_op = symplectic_to_pauliword(symmetry_op_sympletic)

        sym_op_qNos_Paulis, sym_op_coeff = zip(*symmetry_op.terms.items())
        sym_op_qNos, sym_op_Paulis = zip(*sym_op_qNos_Paulis[0])

        # find unique qubit index of independent commuting generator!
        unqiue_indices = list(set(sym_op_qNos).intersection(unqiue_qubit_indices))
        unqiue_index = unqiue_indices[0]

        # pauli of Gi in unique position
        pauli = sym_op_Paulis[sym_op_qNos.index(unqiue_index)]

        sigma_op = QubitOperator(f'{anti_comm_pauli[pauli]}{unqiue_index}', 1)

        # U = const * (symmetry_op + sigma_op)
        list_clifford_like_rot.append(U_op(Gi= symmetry_op, Si=sigma_op))

    return list_clifford_like_rot


# def Apply_U_tapering_symmetry_rotations(pauli_hamiltonian: QubitOperator, list_of_U_op_rotations, n_qubits: int):
#
#     # TODO: can probably do this symbolically instead!
#     # note need to keep track of rotated H (this is where was and is now fixed... note too sure about
#     # commutativity check using sympletic form of full H! Maybe check generators and U_ops! May be okay as indpedent.
#
#     if isinstance(pauli_hamiltonian, QubitOperator):
#         pauli_hamiltonian = list(pauli_hamiltonian)
#
#     H_sympletic = build_G_matrix(pauli_hamiltonian, n_qubits)
#     zero_mat = np.zeros((n_qubits, n_qubits), dtype=int)
#     I_mat = np.eye(n_qubits, dtype=int)
#     J_symplectic = np.block([
#         [zero_mat, I_mat],
#         [I_mat, zero_mat]
#     ])
#     HT_J = np.transpose(H_sympletic) @ J_symplectic
#
#     rotated_H = deepcopy(pauli_hamiltonian)
#     for named_op in list_of_U_op_rotations:
#         G_i = named_op.Gi
#         single_qubit_pauli = named_op.Si
#
#         sympletic_single_q = pauliword_to_symplectic(n_qubits, single_qubit_pauli)
#         sympletic_commutativity = HT_J@sympletic_single_q
#         commutativity = (sympletic_commutativity+1)%2
#
#         for ind, comm_flag in enumerate(commutativity):
#             pauli_h = rotated_H[ind]
#             if comm_flag ==1:
#                 rotated_H[ind] = pauli_h
#             else:
#                 rotated_H[ind] = G_i*single_qubit_pauli*pauli_h
#
#     return rotated_H

def Apply_U_tapering_symmetry_rotations(pauli_hamiltonian: QubitOperator, list_of_U_op_rotations):
    # symbolic commutativity check!

    if isinstance(pauli_hamiltonian, list):
        pauli_hamiltonian = reduce(lambda x,y:x+y, pauli_hamiltonian)

    rotated_H = deepcopy(pauli_hamiltonian)
    for named_op in list_of_U_op_rotations:
        temp_H_rot = QubitOperator()
        G_i = named_op.Gi
        single_qubit_pauli = named_op.Si

        for op in rotated_H:
            if op * single_qubit_pauli == single_qubit_pauli * op:
                temp_H_rot += op
            else:
                temp_H_rot += G_i*single_qubit_pauli*op
        rotated_H = temp_H_rot

    return rotated_H


def get_rotated_operator_and_generators(pauli_hamiltonian: QubitOperator) -> Tuple[QubitOperator, list, list, list]:
    """
    Function maps list of symmetry operators to single qubit Pauli X operators using Clifford rotations

    Args:
        pauli_hamiltonian (QubitOperator): QubitOperator to perform tapering on
    Returns:
        H_rotated (QubitOperator): QubitOperator that has been rotated according to symmetry operators
        symmetry_generators (list): list of symmetry operators in pauli_hamiltonian
        single_X_generators (list): list of symmetry operators that have been rotated to be single Pauli X terms
        clifford_rotations (list): list of cliford rotations used to reduce all symmetry operators to single Pauli X terms
    """
    if isinstance(pauli_hamiltonian, list):
        pauli_hamiltonian = reduce(lambda x,y:x+y, pauli_hamiltonian)

    n_qubits = count_qubits(pauli_hamiltonian)

    G_mat= build_G_matrix(list(pauli_hamiltonian), n_qubits)
    E_mat = build_E_matrix(G_mat)

    E_tilda = gf2_gaus_elim(E_mat)

    # remove rows of zeros!
    E_tilda = E_tilda[~np.all(E_tilda == 0, axis=1)]

    basis_vecs_of_kernel_for_Etilda = gf2_basis_for_gf2_rref(E_tilda)

    U_clifford_rotations = Global_commuting_terms_reduction_to_single_Q_ops(basis_vecs_of_kernel_for_Etilda)
    H_rotated  = Apply_U_tapering_symmetry_rotations(pauli_hamiltonian, U_clifford_rotations)

    return H_rotated, U_clifford_rotations


def find_all_sectors(rotated_hamiltonian: QubitOperator, list_of_U_op_rotations: list) -> List:
    """
    Function takes in symmetry rotated Hamiltonian and single qubit X symmetry generators and returns all the
    tapered Hamiltonians ordered by increasing eigenvalue.

    Note this function performs diagonlisation and returns 2^num_generators Hamiltonians

    Args:
        rotated_hamiltonian (QubitOperator): QubitOperator that has been rotated according to symmetry operators
    Returns:
        H_rotated (QubitOperator): QubitOperator that has been rotated according to symmetry operators
        symmetry_generators (list): list of symmetry operators that have been rotated to be single Pauli X terms
    """
    symmetry_generator_qubit_indices = set()
    for u_op in list_of_U_op_rotations:
        single_qubit_generator = u_op.Si
        for Pauliword, coeff in single_qubit_generator.terms.items():
            qNos, _ = zip(*Pauliword)
            symmetry_generator_qubit_indices.add(*qNos)

    # build dictionary to map qubits to new indices
    n_qubits = count_qubits(rotated_hamiltonian)
    qubit_inds_kept = set(range(n_qubits)).difference(symmetry_generator_qubit_indices)
    qubit_relabel = dict(zip(qubit_inds_kept, range(len(qubit_inds_kept))))
    all_possible_H = []
    # iterate over all possible 2^n_generator +1/-1 measurement assignments
    for measurement_assignment in itertools.product([+1, -1], repeat=len(symmetry_generator_qubit_indices)):

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
        all_possible_H.append((H_reduced, generator_ind_and_meas_val))

    # find eig vals of all possible terms
    H_red_eig_vals=[]
    for H_red, _ in all_possible_H:
        sparse_H_red = get_sparse_operator(H_red)
        if sparse_H_red.shape[0]< 128:
            eigvals, eigvecs = np.linalg.eigh(sparse_H_red.todense())
            min_eigval = min(eigvals)
        else:
            min_eigval = eigsh(sparse_H_red, k = 1, which = 'SA')[0][0]

        H_red_eig_vals.append(min_eigval)

    return sorted(list(zip(H_red_eig_vals, all_possible_H)), key=lambda x:x[0])


def find_sector_using_input_state(rotated_hamiltonian: QubitOperator, list_of_U_op_rotations: list,
                                       qubit_state: np.array, check_eigenvalue_by_ind: int = None) -> QubitOperator:
    """
    Function that returns tapered Hamiltonian according to given qubit_state.

    Function gets the eigenvalues of symmetry_generators_pre_rotatation measured on the input qubit state
    and uses these to define the measurements of the symmetry_generators (rather than brute force searching over
    all possible measurement assignments to them).

    Note check_correct performs diagonlisation of the full rotated_hamiltonian (contains all qubits of problem)
    and compares minimum eigenvalue to tapered H (fewer qubits)
    This can be expensive to check!

    Args:
        rotated_hamiltonian (QubitOperator): QubitOperator that has been rotated according to symmetry operators
        symmetry_generators_pre_rotatation (list): List of symmetry generators before clifford rotations
        qubit_state (np.array): qubit state that defines eigenvalues of symmetry_generators_pre_rotatation
        symmetry_generators (list): List of single qubit X symmetry generators
        check_eigenvalue_by_ind (int): optional int to compare eigenvalue ith (starting from the lowest) - defined by input index
        - to that of the rotated_hamiltonian (no qubits removed). Ground state would by 0, first excited state 1 ... etc
    Returns:
        H_reduced (QubitOperator): ground state tapered Hamiltonian
    """
    qubit_state_Z_mesaure = -1*((2*qubit_state) - 1) # plus and minus one for Z measurement
    single_sig_q_ind_and_meas_val = {}
    single_s_symmetry_generator_qubit_indices = set()
    for sym_op_ind, u_op in enumerate(list_of_U_op_rotations):
        G_symm_op = u_op.Gi
        sym_gen_sig_qno, coeff = zip(*G_symm_op.terms.items())
        sym_qno, sym_paulis = zip(*sym_gen_sig_qno[0])
        if not ''.join(sym_paulis) == 'Z' * len(sym_qno):
            raise ValueError(f'Currently function only works for Z generators! {G_symm_op} not valild')

        single_q_op, coeff = zip(*u_op.Si.terms.items())
        single_q_op_qNos, single_q_op_Paulis = zip(*single_q_op[0])
        unique_q_index = single_q_op_qNos[0]
        single_s_symmetry_generator_qubit_indices.add(unique_q_index)

        single_sig_q_ind_and_meas_val[unique_q_index] = np.prod(np.take(qubit_state_Z_mesaure, sym_qno)) * coeff[0]
        # print(f'<{G_symm_op}> = {np.prod(np.take(qubit_state_Z_mesaure, sym_qno))}, for: {qubit_state}')

    # build dictionary to map qubits to new indices
    n_qubits = count_qubits(rotated_hamiltonian)
    qubit_inds_kept = set(range(n_qubits)).difference(single_s_symmetry_generator_qubit_indices)
    qubit_relabel = dict(zip(qubit_inds_kept, range(len(qubit_inds_kept))))

    H_reduced = QubitOperator()
    for h_op in rotated_hamiltonian:
        if not list(h_op.terms.keys())[0]:
            H_reduced += h_op
        else:
            for Pauliword, coeff in h_op.terms.items():
                new_coeff = coeff
                qNos, paulis = zip(*Pauliword)
                for qNo in set(qNos).intersection(single_s_symmetry_generator_qubit_indices):
                    new_coeff = new_coeff*single_sig_q_ind_and_meas_val[qNo]

            qNos_reduced = set(qNos).difference(single_s_symmetry_generator_qubit_indices)
            Pauliword_reduced = ' '.join([f'{Pauli}{qubit_relabel[qNo]}'for qNo, Pauli in zip(qNos, paulis)
                                          if qNo in qNos_reduced])

            H_reduced += QubitOperator(Pauliword_reduced, new_coeff)

    if check_eigenvalue_by_ind:
        sparse_H_full = get_sparse_operator(rotated_hamiltonian)
        if sparse_H_full.shape[0] < 128:
            eigenValues, eigenVectors = np.linalg.eigh(sparse_H_full.todense())
            idx = eigenValues.argsort()
            eigvals = eigenValues[idx]
            # eigvecs = eigenVectors[:, idx]
            min_eigval_full = eigvals[check_eigenvalue_by_ind]
        else:
            min_eigval_full = eigsh(sparse_H_full, k=(check_eigenvalue_by_ind+1), which='SA')[0][0]

        sparse_H_red = get_sparse_operator(H_reduced)
        if sparse_H_red.shape[0]< 128:
            eigenValues, eigenVectors = np.linalg.eigh(sparse_H_red.todense())
            idx = eigenValues.argsort()
            eigvals = eigenValues[idx]
            # eigvecs = eigenVectors[:, idx]
            min_eigval = eigvals[check_eigenvalue_by_ind] #min(eigvals)
        else:
            min_eigval = eigsh(sparse_H_red, k=(check_eigenvalue_by_ind+1), which = 'SA')[0][0]

        if not np.isclose(min_eigval_full, min_eigval):
            print(min_eigval_full)
            print(min_eigval)
            raise ValueError('wrong sector found')

    return H_reduced


def get_ground_tapered_H_using_HF_state(pauli_hamiltonian: QubitOperator, HF_qubit_state: np.array,
                                        check_tapering:bool = False) -> QubitOperator:
    """
    Function that returns tapered Hamiltonian, where HF ground state is used to define the Sector

    Args:
        pauli_hamiltonian (QubitOperator): QubitOperator that has been rotated according to symmetry operators
        HF_qubit_state (np.array): ket as numpy array representing qubit state used to define sector (note
                                   symmetry generators are measured on this state to define fixed eigenvalues)
        check_tapering (bool): optional flag to check ground state of tapered H with original H (performs sparse
                                diagonalization - can be expensive!)
    Returns:
        H_tapered (QubitOperator): ground state tapered Hamiltonian
    """
    H_rotated, U_clifford_rotations = get_rotated_operator_and_generators(pauli_hamiltonian)

    if check_tapering:
        H_tapered = find_sector_using_input_state(H_rotated, U_clifford_rotations, HF_qubit_state,
                                                  check_eigenvalue_by_ind=0) # <- zero th
    else:
        H_tapered = find_sector_using_input_state(H_rotated, U_clifford_rotations, HF_qubit_state,
                                                  check_eigenvalue_by_ind=None)

    return H_tapered


if __name__ ==  '__main__':

    # spin up then spin down
    JW_ket = np.eye(2**4)[int('1010',2)].reshape(16,1)
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
    E_tilda = gf2_gaus_elim(E)
    E_tilda = E_tilda[~np.all(E_tilda == 0, axis=1)] # remove rows of zeros!

    basis_vecs_of_kernel_for_Etilda = gf2_basis_for_gf2_rref(E_tilda)

    for stabilizer in basis_vecs_of_kernel_for_Etilda:
        Pword = symplectic_to_pauliword(stabilizer)
        print(Pword)

    U_list = Global_commuting_terms_reduction_to_single_Q_ops(basis_vecs_of_kernel_for_Etilda)
    H_rotated = Apply_U_tapering_symmetry_rotations(H, U_list)

    H_full = get_sparse_operator(reduce(lambda x,y:x+y, H))
    H_full_rotated = get_sparse_operator(reduce(lambda x, y: x + y, H_rotated))
    eigvals, eigvecs = eigh(H_full.todense())
    eigvals2, eigvecs2 = eigh(H_full_rotated.todense())
    print(np.allclose(eigvals, eigvals2))

    full_spectrum_tapered = find_all_sectors(H_rotated, U_list)

    H_tapered_sector_by_state = find_sector_using_input_state(H_rotated, U_list, JW_array, check_eigenvalue_by_ind=0)

