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
    n_qubits = count_qubits(pauli_hamiltonian)

    G_mat= build_G_matrix(list(pauli_hamiltonian), n_qubits)
    E_mat = build_E_matrix(G_mat)

    E_tilda = gf2_gaus_elim(E_mat)

    # remove rows of zeros!
    E_tilda = E_tilda[~np.all(E_tilda == 0, axis=1)]

    basis_vecs_of_kernel_for_Etilda = gf2_basis_for_gf2_rref(E_tilda)

    symmetry_generators = []
    for stabilizer in basis_vecs_of_kernel_for_Etilda:
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
        check_correct (bool): optional flag to compare minimum eigenvalue of rotated_hamiltonian with tapered H
                              note this should be the same for function to have been succcessful
    Returns:
        H_reduced (QubitOperator): ground state tapered Hamiltonian
    """
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
        if not list(h_op.terms.keys())[0]:
            H_reduced += h_op
        else:
            for Pauliword, coeff in h_op.terms.items():
                new_coeff = coeff
                qNos, paulis = zip(*Pauliword)
                for qNo in set(qNos).intersection(symmetry_generator_qubit_indices):
                    try:
                        new_coeff = new_coeff*generator_ind_and_meas_val[qNo]
                    except:
                        print(generator_ind_and_meas_val)
                        print(qNo)
                        print(symmetry_generators_pre_rotatation)
                        raise ValueError()

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
    (H_rotated, symmetry_generators,
    single_X_generators, clifford_rotations) = get_rotated_operator_and_generators(pauli_hamiltonian)

    H_tapered = find_ground_sector_using_input_state(H_rotated, symmetry_generators, HF_qubit_state,
                                                     single_X_generators, check_correct=check_tapering)

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

    symmetry_generators = []
    for stabilizer in basis_vecs_of_kernel_for_Etilda:
        Pword = symplectic_to_pauliword(stabilizer)
        print(Pword)
        symmetry_generators.append(Pword)

    single_X_generators, cliff_rotations = Clifford_operator_reduction_to_X(symmetry_generators)

    H_rotated = deepcopy(H)

    for rot in cliff_rotations:
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


    all_tapered_H = find_sector_brute_force(H_rotated, single_X_generators)
    print(all_tapered_H)

    H_tapered_ground = find_ground_sector_using_input_state(H_rotated, symmetry_generators,
                                                            JW_array, single_X_generators,check_correct = True)

    H_tapered_ground_mat = get_sparse_operator(reduce(lambda x,y:x+y, H_tapered_ground))
    eigvals3, eigvecs3 = eigh(H_tapered_ground_mat.todense())
    print()
    print('ground H tapered:')
    print(min(eigvals3))
    print(H_tapered_ground)

    H_tap = get_ground_tapered_H_using_HF_state(reduce(lambda x,y:x+y, H), JW_array, check_tapering=True)
    print(H_tap == H_tapered_ground)



