from functools import cached_property
from cirq import Pauli
import numpy as np


def symplectic_to_string(symp_vec) -> str:
    """
    Returns string form of symplectic vector defined as (X | Z)

    Args:
        symp_vec (array): symplectic Pauliword array

    Returns:
        Pword_string (str): String version of symplectic array

    """

    n_qubits = len(symp_vec) // 2

    X_block = symp_vec[:n_qubits]
    Z_block = symp_vec[n_qubits:]

    Y_loc = ((X_block + Z_block) == 2).astype(bool)
    X_loc = np.bitwise_xor(Y_loc, X_block).astype(bool)
    Z_loc = np.bitwise_xor(Y_loc, Z_block).astype(bool)

    char_aray = np.array(list('I' * n_qubits), dtype=str)

    char_aray[Y_loc] = 'Y'
    char_aray[X_loc] = 'X'
    char_aray[Z_loc] = 'Z'

    Pword_string = ''.join(char_aray)

    return Pword_string


# class Pauliword():

#     def __init__(self, paulis, coeffs):
#         self.paulis = paulis
#         self._valid_pauliword()
#         self.coeffs = coeffsop_list = ['IZXY', 'YYXX']
coeff = [-1,1j]
op1 = Pauliword(op_list, coeff)
op2 =  Pauliword(op1.symp_matrix, coeff)


#         self._init_symplectic()

#     def __str__(self):
#         return f'{self.coeff} {self.pauli_string}'

#     def _valid_pauliword(self):
#         """
#         check to see if Pauli is made up of single qubit Pauli operators only
#         """
#         paulis = set(['X', 'Z', 'Y', 'I'])
#         assert (set(self.pauli_string).issubset(paulis)), 'pauliword must only contain X,Y,Z,I terms'

#     def _init_symplectic(self):
#         symp = np.zeros(2 * self.n_qubits, dtype=int)

#         char_aray = np.array(list(self.pauli_string), dtype=str)
#         X_loc = (char_aray == 'X')
#         Z_loc = (char_aray == 'Z')
#         Y_loc = (char_aray == 'Y')

#         symp[:self.n_qubits] += X_loc
#         symp[self.n_qubits:] += Z_loc
#         symp[:self.n_qubits] += Y_loc
#         symp[self.n_qubits:] += Y_loc

#         self.vec = symp

#         self.X_block = symp[:self.n_qubits]
#         self.Z_block = symp[self.n_qubits:]
#         self.Y_count = sum(Y_loc)
    
#     def _init_counts_from_symplectic(self):
#         self.X_block = self.vec[:self.n_qubits]
#         self.Z_block = self.vec[self.n_qubits:]
#         self.Y_count = sum(((self.X_block + self.Z_block) == 2))


#     def commutes(self, Pword) -> bool:
#         # checks qubit number is same
#         assert (self.n_qubits == Pword.n_qubits), 'Pauliwords defined for different number of qubits'

#         gf2_comm = (self.Z_block @ Pword.X_block + self.X_block @ Pword.Z_block) % 2
#         return not bool(gf2_comm)

#     def signless_multiplication_symp(self, Pword) -> np.array:
#         pauli_mult_symp_vec = np.bitwise_xor(self.vec, Pword.vec)
#         return ppauli_vecauli_mult_symp_vec

#     def signless_multiplication_pword(self, Pword):
#         pauli_mult_symp_vec = self.signless_multiplication_symp(Pword)
#         new_p_string = symplectic_to_string(pauli_mult_symp_vec)
#         return Pauliword(new_p_string, 1)

    # def __mul__(self, Pword):
    #     signless_prod_Pword = self.signless_multiplication_pword(Pword)
    #     num_sign_flips = np.sum(np.bitwise_xor(self.X_block, Pword.Z_block))
    #     sign_change = (-1) ** num_sign_flips

    #     # mapping from sigma to tau representation
    #     full_Y_count = self.Y_count + Pword.Y_count
    #     sigma_tau_compensation = (-1j) ** full_Y_count

    #     # back from tau to sigma (note uses output Pword)
    #     tau_sigma_compensation = (1j) ** signless_prod_Pword.Y_count

    #     phase_mod = sign_change * sigma_tau_compensation * tau_sigma_compensation

    #     overall_phase = phase_mod * self.coeff * Pword.coeff

    #     P_out = Pauliword(signless_prod_Pword.pauli_string, overall_phase)
    #     return P_out
# symp_matrix = np.zeros((n_rows, 2 * self.n_qubits), dtype=int)
#     def __sub__(self):
#         # TODO: function for subtraction!
#         pass
#     def __eq__(self, Pword):
#         """
#         checks equivlance of current Pauliword with different Pauliword
#         aka defines the  == operation
#         """
#         if (self.pauli_string, self.coeff) == (Pword.pauli_string, Pword.coeff):
#             return True
#         else:
#             return False


# class PauliSentence():

#     def __init__(self, list_pauliwords, n_qubits):
#         self.list_pauliwords = list_pauliwords
#         self.n_qubits = n_qubits

#     def _init_symplectic_matrix(self):

#         self.symp_matrix = np.zeros((len(self.list_pauliwords), 2*self.n_qubits), dtype=int)
#         self.coeff_vec = np.zeros((len(self.list_pauliwords), 1), dtype=complex)
#         self._comm_matrix = np.zeros((len(self.list_pauliwords), 2*self.n_qubits), dtype=int)
#         self._sign_matrix = np.zeros((len(self.list_pauliwords), 2 * self.n_qubits), dtype=int)
#         for ind, Pword in enumerate(self.list_pauliwords):
#             self.symp_matrix[ind] = Pword.vec
#             self.coeff_vec[ind] = Pword.coeff

#             self._comm_matrix = np.hstack((Pword.Z_block, Pword.X_block))
#             self._sign_matrix[ind, :self.n_qubits] = Pword.X_block
#         return None


#     def commutes(self, alt_linear_combination_pauliwords):
#         pass



# from qondense.utils.symplectic_toolkit import *
from copy import deepcopy
from typing import List, Union, Dict
from scipy import sparse
from functools import reduce

class Pauliword:
    """ Class to represent an operator defined over the Pauli group 
    in the symplectic form. The internal symplectic matrix is stored
    in the compressed sparse column (CSC) form to enable fast matrix 
    operations.
    """
    def __init__(self, 
            operator: Union[List[str], np.array], coeff_list: List[complex]
        ):
        """ When the class is first initialized it is easiest to provide
        the operator stored as a dictionary where keys are strings representing
        Pauli operators and values coefficients. The operator may also be given
        as a string or list of strings, in which case the coefficients will be 
        set to 1. However, in the interest of efficiency, whenever a method creates
        a new QubitOp instance it will instead specify the operator in the sparse
        form, with a vector of coefficients stored as an array. This way we are
        not constantly convertng back and forth between the dictionary and 
        symplectic represenations.
        """
        self.coeff_vec = np.asarray(coeff_list, dtype=complex)

        if isinstance(operator, list):
            self._init_from_paulistring_list(operator)
        elif isinstance(operator, np.ndarray):
            if len(operator.shape)==1:
                operator = operator.reshape([1, len(operator)])

            self.symp_matrix = operator
            self.n_qubits = self.symp_matrix.shape[1]//2
        else:
            raise ValueError(f'unkown operator type: must be dict or np.array: {type(operator)}')

        assert(self.symp_matrix.shape[0]==len(self.coeff_vec)), 'coeff list and Pauliwords not same length'

        assert(set(np.unique(self.symp_matrix)).issubset({0,1})), 'symplectic array not defined with 0 and 1 only'
        self.X_block = self.symp_matrix[:, :self.n_qubits]
        self.Z_block = self.symp_matrix[:, self.n_qubits:]

    def _init_from_paulistring_list(self, operator_list):
        n_rows = len(operator_list)
        self.n_qubits = len(operator_list[0])

        self.symp_matrix = np.zeros((n_rows, 2 * self.n_qubits), dtype=int)
        for row_ind, pauli_str in enumerate(operator_list):
            
            assert(len(pauli_str)==self.n_qubits), 'operator defined on differing qubit numbers'
            assert (set(pauli_str).issubset({'I', 'X', 'Y', 'Z'})), 'pauliword must only contain X,Y,Z,I terms'

            char_aray = np.array(list(pauli_str), dtype=str)
            X_loc = (char_aray == 'X')
            Z_loc = (char_aray == 'Z')
            Y_loc = (char_aray == 'Y')

            self.symp_matrix[row_ind, :self.n_qubits] += X_loc
            self.symp_matrix[row_ind,self.n_qubits:] += Z_loc
            self.symp_matrix[row_ind,:self.n_qubits] += Y_loc
            self.symp_matrix[row_ind,self.n_qubits:] += Y_loc

    def __str__(self) -> str:
        out_string = ''
        for pauli_vec, ceoff in zip(self.symp_matrix, self.coeff_vec):
            p_string = symplectic_to_string(pauli_vec)
            out_string += (f'{ceoff} {p_string} +\n')
        return out_string[:-3]


    def copy(self) -> "Pauliword":
        """ Create a carbon copy of the class instance
        """
        return deepcopy(self)

    @cached_property
    def Y_count(self) -> np.array:
        """ 
        Count the qubit positions of each term set to Pauli Y

        cached_property means this only runs once and then is stored
        as self.Y_count
        """
        # Y_coords = self.X_block + self.Z_block == 2
        Y_coords = np.bitwise_and(self.X_block, self.Z_block)
        return np.array(Y_coords.sum(axis=1))

    def _multiply_single_Pword_phaseless(self,Pword) -> np.array:
        """ performs *phaseless* Pauli multiplication via binary summation 
        of the symplectic matrix. Phase requires additional operations that
        are computed in phase_modification.
        """
        pauli_mult_phaseless = np.bitwise_xor(self.symp_matrix, Pword.symp_matrix)
        return Pauliword(pauli_mult_phaseless, np.ones(pauli_mult_phaseless.shape[0]))
    
    def _multiply_single_Pword(self, Pword):
        
        signless_prod_Pword = self._multiply_single_Pword_phaseless(Pword)

        # counts ZX mismatches
        assert(Pword.Z_block.shape[0]==1), 'not single Pauliword'
        num_sign_flips = np.sum(np.bitwise_and(self.X_block, Pword.Z_block),
                               axis=1)

        sign_change = (-1) ** num_sign_flips

        # mapping from sigma to tau representation
        full_Y_count = self.Y_count + Pword.Y_count
        sigma_tau_compensation = (-1j) ** full_Y_count

        # back from tau to sigma (note uses output Pword)
        tau_sigma_compensation = (1j) ** signless_prod_Pword.Y_count

        phase_mod = sign_change * sigma_tau_compensation * tau_sigma_compensation

        new_coeff_vec = phase_mod * self.coeff_vec * Pword.coeff_vec

        P_out = Pauliword(signless_prod_Pword.symp_matrix, new_coeff_vec)
        return P_out

    def cleanup(self):
        # order lexicographically and take difference between adjacent rows
        term_ordering = np.lexsort(self.symp_matrix.T)
        diff_adjacent = np.diff(self.symp_matrix[term_ordering], axis=0)
        # the unique terms are those which are non-zero
        mask_unique_terms = np.append(True, ~np.all(diff_adjacent==0, axis=1))

        # determine the inverse mapping that combines duplicate terms
        inverse_index = np.zeros_like(term_ordering)
        inverse_index[term_ordering] = np.cumsum(mask_unique_terms) - 1
        
        # drop duplicate terms
        simplified_terms = self.symp_matrix[term_ordering][mask_unique_terms]
        # sum over the coefficients of duplicated terms
        # note np.bincount doesn't like complex numbers
        weights = self.coeff_vec
        cf_real = np.bincount(inverse_index, weights = weights.real)
        cf_imag = np.bincount(inverse_index, weights = weights.imag)
        # print(cf_real, np.array(cf_imag)*1j)
        simplified_coeff = cf_real+cf_imag*1j
        return Pauliword(simplified_terms, simplified_coeff)

    def __add__(self, Pword):

        P_symp_mat_new = np.vstack((self.symp_matrix, Pword.symp_matrix))
        P_new_coeffs = np.hstack((self.coeff_vec, Pword.coeff_vec)) 

        # cleanup run to remove duplicate rows (Pauliwords)
        P_new = Pauliword(P_symp_mat_new, P_new_coeffs).cleanup()
        
        return P_new


    def __mul__(self, Pword):

        P_updated_list =[]
        for ind in range(Pword.symp_matrix.shape[0]):
            Pvec_single = Pword.symp_matrix[ind]
            coeff_single= Pword.coeff_vec[ind]
            Pword_temp = Pauliword(Pvec_single, [coeff_single])
            # print(Pword_temp)
            P_new = self._multiply_single_Pword(Pword_temp)
            P_updated_list.append(P_new)

        # for op in P_updated_list:
        #     print(op)

        P_final = reduce(lambda x,y: x+y, P_updated_list)
        return P_final

    def commutes(self, Pword) -> bool:
        # checks qubit number is same
        # TODO: fix this function!
        assert (self.n_qubits == Pword.n_qubits), 'Pauliwords defined for different number of qubits'
        # gf2_comm = np.bitwise_xor(np.sum(np.bitwise_and(self.Z_block, Pword.X_block), axis=1),
        #                           np.sum(np.bitwise_and(self.X_block, Pword.Z_block), axis=1))
        
        return not bool(gf2_comm)

    # def basis_reconstruction(self, 
    #         operator_basis: List[str]
    #     ) -> np.array:
    #     """ simultaneously reconstruct everX_blockis elements required to reconstruct
    #     the corresponding term in the operator.
    #     """
    #     dim = len(operator_basis)
    #     basis_symp_csc = self.reform(operator_basis)._symp_csc
    #     basis_op_stack = sparse.vstack([basis_symp_csc, self._symp_csc])
    #     ham_reconstruction = gf2_gaus_elim(basis_op_stack.toarray().T)[:dim,dim:].T

    #     return ham_reconstruction


    # def symplectic_inner_product(self, 
    #         aux_paulis: Union[str, List[str], Dict[str, float], np.array, "QubitOp"], 
    #         sip_type = 'full'
    #     ) -> np.array:
    #     """ Method to calculate the symplectic inner product of the represented
    #     operator with one (or more) specified pauli operators, .

    #     sip_type allows one to choose whether the inner product is: 
    #     - full, meaning it computes commutation properties, or... 
    #     - half, which computes sign flips for Pauli multiplication
    #     """
    #     if sip_type == 'full':
    #         Omega = self.symform_csc
    #     elif sip_type == 'half':
    #         Omega = self.half_symform_csc
    #     else:
    #         raise ValueError('Accepted values for sip_type are half or full')
        
    #     aux_paulis = self.reform(aux_paulis)
    #     sparse_inner_product = self._symp_csc @ Omega @ aux_paulis._symp_csc.transpose()
    #     sparse_inner_product.data %= 2 # effects modulo 2 in sparse form

    #     return sparse_inner_product


    # def commutes_with(self, 
    #         aux_paulis: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> np.array:
    #     """ Returns an array in which:
    #     - 0 entries denote commutation and
    #     - 1 entries denote anticommutation
    #     """
    #     return self.symplectic_inner_product(aux_paulis=aux_paulis)

    
    # def adjacency_matrix(self) -> np.array:
    #     """ Checks commutation of the represented operator with itself
    #     """
    #     return self.commutes_with(aux_paulis=self._symp_csc.toarray())


    # def sign_difference(self, 
    #         aux_paulis: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> np.array:
    #     """ symplectic inner product but with a modified syplectic form.
    #     This keeps track of sign flips resulting from Pauli multiplication
    #     but disregards complex phases (we account for this elsewhere).
    #     """
    #     return self.symplectic_inner_product(aux_paulis=aux_paulis, sip_type='half')


    # def phase_modification(self, 
    #         source_pauli: Union[str, List[str], Dict[str, float], np.array, "QubitOp"], 
    #         target_pauli: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> np.array:
    #     """ compensates for the phases incurred through Pauli multiplication
    #     implemented as per https://doi.org/10.1103/PhysRevA.68.042318

    #     outputs a vector of phases to multiply termwise with the coefficient vector
    #     """
        
    #     sign_exp = self.sign_difference(source_pauli).toarray()
    #     sign = (-1)**sign_exp
    #     Y_count = self.count_pauli_Y() + source_pauli.count_pauli_Y()
    #     sigma_tau_compensation = (-1j)**Y_count # mapping from sigma to tau representation
    #     tau_sigma_compensation = (1j)**target_pauli.count_pauli_Y() # back from tau to sigma
    #     phase_mod = sign*sigma_tau_compensation*tau_sigma_compensation # the full phase modification
    #     return phase_mod



    # def multiply_by_pauli(self, 
    #         pauli: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> "QubitOp":
    #     """ computes right-multiplication of the internal operator 
    #     by some single pauli operator... *with* phases
    #     """
    #     pauli = self.reform(pauli)
    #     assert(pauli.n_terms==1) # must be a single pauli operator

    #     # perform the pauli multiplication disregarding phase
    #     phaseless_product = self._multiply_by_pauli(pauli)
    #     phase_mod = self.phase_modification(pauli, phaseless_product)
    #     new_cfvec = self.cfvec*pauli.cfvec*phase_mod
        
    #     return QubitOp(symp_rep=phaseless_product._symp_csc, coeffvec=new_cfvec)


    # def multiply_by_operator(self, 
    #         operator: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> "QubitOp":
    #     """ computes right-multiplication of the internal operator 
    #     by some other operator that may contain arbitrarily many terms
    #     """
    #     operator = self.reform(operator)
    #     pauli_products = []
    #     cfvec_products = []
    #     for pauli, coeff in operator._dict().items():
    #         product = self.multiply_by_pauli({pauli:coeff})
    #         pauli_products.append(product._symp())
    #         cfvec_products.append(product.cfvec)

    #     clean_terms, clean_coeff = cleanup_symplectic(
    #             terms=np.vstack(pauli_products), 
    #             coeff=np.vstack(cfvec_products)
    #         )
    #     return QubitOp(symp_rep=clean_terms, coeffvec=clean_coeff) 
            

    # def _rotate_by_pauli(self, 
    #         pauli: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> "QubitOp":
    #     """ Performs (H Omega vT \otimes v) \plus H where H is the internal operator, 
    #     Omega the symplectic form and v the symplectic representation of pauli
    #     """
    #     pauli = self.reform(pauli)
    #     assert(pauli.n_terms==1) # must be a single pauli operator

    #     commutes = self.commutes_with(pauli)
    #     rot_where_commutes = sparse.kron(commutes, pauli._symp_csc)
    #     phaseless_rotation = self._symp_csc + rot_where_commutes
    #     phaseless_rotation.data %= 2 # modulo 2
    #     phaseless_rotation = QubitOp(symp_rep = phaseless_rotation,
    #                                 coeffvec = self.cfvec)
    #     return phaseless_rotation, commutes.toarray()

    
    # def rotate_by_pauli(self, 
    #         pauli: Union[str, List[str], Dict[str, float], np.array, "QubitOp"], 
    #         angle: float = np.pi/2, 
    #         clifford_flag: bool = True
    #     ) -> "QubitOp":
    #     """ Let R(t) = e^{i t/2 Q}, then one of the following can occur:
    #     R(t) P R^\dag(t) = P when [P,Q] = 0
    #     R(t) P R^\dag(t) = cos(t) P + sin(t) (-iPQ) when {P,Q} = 0
        
    #     This operation is Clifford when t=pi/2, since cos(pi/2) P - sin(pi/2) iPQ = -iPQ.
    #     For t!=pi/2 an increase in the number of terms can be observed (non-Clifford unitary).
    #     """
    #     pauli = self.reform(pauli)
    #     phaseless_rotation, commutes = self._rotate_by_pauli(pauli)
    #     phase_mod = self.phase_modification(pauli, phaseless_rotation)
    #     # zero out phase mod where term commutes
    #     phase_mod = phase_mod*commutes
    #     commutes_inv = np.array(commutes==0, dtype=int) #inverted commutes vector
    #     if clifford_flag:
    #         # rotates by pi/2 regardless of specified angle
    #         # set clifford=False to change this behaviour
    #         phase_mod = commutes_inv - phase_mod*1j # -1j comes from rotation derivation above
    #         new_cfvec = self.cfvec*pauli.cfvec*phase_mod
    #         return QubitOp(symp_rep=phaseless_rotation._symp_csc, coeffvec=new_cfvec)
    #     else:
    #         phase_mod = commutes_inv - np.sin(angle)*phase_mod*1j
    #         new_cfvec = self.cfvec*pauli.cfvec*phase_mod
    #         # when non-Clifford the anti-commuting terms can split in two under rotation:
    #         extra_ops = self._symp_csc[np.where((commutes==1).T[0])]
    #         extra_cfv = self.cfvec[np.where((commutes==1).T[0])]*pauli.cfvec*np.cos(angle)
    #         # remove and sum coefficients of duplicate terms arising from this split
    #         clean_terms, clean_coeff = cleanup_symplectic(
    #             terms=sparse.vstack([phaseless_rotation._symp_csc, extra_ops]).toarray(), 
    #             coeff=np.vstack([new_cfvec, extra_cfv])
    #         )
    #         return QubitOp(symp_rep=clean_terms, coeffvec=clean_coeff)
            
    
    # def perform_rotations(self, 
    #         rotation_list:List[Tuple[str,float,bool]]
    #     ) -> "QubitOp":
    #     """ Allows one to perform a list of rotations sequentially,
    #     stored in the form [(pauli, angle, clifford_flag), ...]
    #     """
    #     op_copy = self.copy()
    #     for pauli, angle, clifford_flag in rotation_list:
    #         op_copy = op_copy.rotate_by_pauli(pauli,angle,clifford_flag)
    #     return op_copy


    # ###### BELOW THIS POINT SOON TO BE DEPRECATED #######
    # ###### Left for the purposes of checking      #######

    # def _dictionary_rotation(self, 
    #                         pauli_rot:str, 
    #                         angle:float=None,
    #                         clifford_flag:bool=True
    #                         )->Dict[str, float]:

    #     ## determine possible Paulis in image to avoid if statements
    #     #pauli_rot_symp_csc = pauli_to_symp_csclectic(pauli_rot)
    #     #I = np.eye(2*self.n_qubits, 2*self.n_qubits)
    #     #OmegaPtxP = np.outer(self.symform @ pauli_rot_symp_csc.T, pauli_rot_symp_csc)
    #     #P_mult_mat = (I+OmegaPtxP) % 2
    #     ## determine Pauli terms
    #     #RQRt = (self._symp_csc @ P_mult_mat) % 2
    #     #poss_ops = np.concatenate((self._symp_csc, RQRt))
    #     #poss_ops = dictionary_operator(
    #     #                                poss_ops, 
    #     #                                np.array([[0] for i in range(len(poss_ops))])
    #     #                                )
    #     def updateymp(self) -> np.array:
    #     """ Get the symplectic matrix in dense form
    #     """
    #     return self._symp_csc.toarray()


    # def _dict(self) -> dict:
    #     """ Get the sparse operator back out as a dictionary.
    #     It is easier to see what the terms are in this represenation. 
    #     """
    #     return dictionary_operator(self._symp(), self.cfvec)


    # def copy(self) -> "QubitOp":
    #     """ Create a carbon copy of the class instance
    #     """
    #     return deepcopy(self)

    
    # def reform(self, 
    #         operator: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> "QubitOp":
    #     """ Funnel the input operator regardless of type into QubitOp
    #     """
    #     if not isinstance(operator, QubitOp):
    #         operator = QubitOp(operator)
    #     return operator.copy()


    # def swap_XZ_blocks(self) -> sparse.csc_matrix:
    #         """ Reverse order of symplectic matrix so that 
    #         Z operators are on the left and X on the right
    #         """
    #         return sparse.hstack((self.Z_block, self.X_block))


    # def count_pauli_Y(self) -> np.array:
    #     """ Count the qubit positions of each term set to Pauli Y
    #     """
    #     Y_coords = self.X_block + self.Z_block == 2
    #     return np.array(Y_coords.sum(axis=1))


    # def basis_reconstruction(self, 
    #         operator_basis: List[str]
    #     ) -> np.array:
    #     """ simultaneously reconstruct everX_blockis elements required to reconstruct
    #     the corresponding term in the operator.
    #     """
    #     dim = len(operator_basis)
    #     basis_symp_csc = self.reform(operator_basis)._symp_csc
    #     basis_op_stack = sparse.vstack([basis_symp_csc, self._symp_csc])
    #     ham_reconstruction = gf2_gaus_elim(basis_op_stack.toarray().T)[:dim,dim:].T

    #     return ham_reconstruction


    # def symplectic_inner_product(self, 
    #         aux_paulis: Union[str, List[str], Dict[str, float], np.array, "QubitOp"], 
    #         sip_type = 'full'
    #     ) -> np.array:
    #     """ Method to calculate the symplectic inner product of the represented
    #     operator with one (or more) specified pauli operators, .

    #     sip_type allows one to choose whether the inner product is: 
    #     - full, meaning it computes commutation properties, or... 
    #     - half, which computes sign flips for Pauli multiplication
    #     """
    #     if sip_type == 'full':
    #         Omega = self.symform_csc
    #     elif sip_type == 'half':
    #         Omega = self.half_symform_csc
    #     else:
    #         raise ValueError('Accepted values for sip_type are half or full')
        
    #     aux_paulis = self.reform(aux_paulis)
    #     sparse_inner_product = self._symp_csc @ Omega @ aux_paulis._symp_csc.transpose()
    #     sparse_inner_product.data %= 2 # effects modulo 2 in sparse form

    #     return sparse_inner_product


    # def commutes_with(self, 
    #         aux_paulis: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> np.array:
    #     """ Returns an array in which:
    #     - 0 entries denote commutation and
    #     - 1 entries denote anticommutation
    #     """
    #     return self.symplectic_inner_product(aux_paulis=aux_paulis)

    
    # def adjacency_matrix(self) -> np.array:
    #     """ Checks commutation of the represented operator with itself
    #     """
    #     return self.commutes_with(aux_paulis=self._symp_csc.toarray())


    # def sign_difference(self, 
    #         aux_paulis: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> np.array:
    #     """ symplectic inner product but with a modified syplectic form.
    #     This keeps track of sign flips resulting from Pauli multiplication
    #     but disregards complex phases (we account for this elsewhere).
    #     """
    #     return self.symplectic_inner_product(aux_paulis=aux_paulis, sip_type='half')


    # def phase_modification(self, 
    #         source_pauli: Union[str, List[str], Dict[str, float], np.array, "QubitOp"], 
    #         target_pauli: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> np.array:
    #     """ compensates for the phases incurred through Pauli multiplication
    #     implemented as per https://doi.org/10.1103/PhysRevA.68.042318

    #     outputs a vector of phases to multiply termwise with the coefficient vector
    #     """
        
    #     sign_exp = self.sign_difference(source_pauli).toarray()
    #     sign = (-1)**sign_exp
    #     Y_count = self.count_pauli_Y() + source_pauli.count_pauli_Y()
    #     sigma_tau_compensation = (-1j)**Y_count # mapping from sigma to tau representation
    #     tau_sigma_compensation = (1j)**target_pauli.count_pauli_Y() # back from tau to sigma
    #     phase_mod = sign*sigma_tau_compensation*tau_sigma_compensation # the full phase modification
    #     return phase_mod


    # def _multiply_by_pauli(self, 
    #         pauli: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> "QubitOp":
    #     """ performs *phaseless* Pauli multiplication via binary summation 
    #     of the symplectic matrix. Phase requires additional operations that
    #     are computed in phase_modification.
    #     """
    #     pauli = self.reform(pauli)
    #     pauli_mult = self._symp_csc + pauli._symp_csc
    #     pauli_mult.data %= 2 # effects modulo 2 in csc form
    #     return QubitOp(symp_rep=pauli_mult, coeffvec=self.cfvec)


    # def multiply_by_pauli(self, 
    #         pauli: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> "QubitOp":
    #     """ computes right-multiplication of the internal operator 
    #     by some single pauli operator... *with* phases
    #     """
    #     pauli = self.reform(pauli)
    #     assert(pauli.n_terms==1) # must be a single pauli operator

    #     # perform the pauli multiplication disregarding phase
    #     phaseless_product = self._multiply_by_pauli(pauli)
    #     phase_mod = self.phase_modification(pauli, phaseless_product)
    #     new_cfvec = self.cfvec*pauli.cfvec*phase_mod
        
    #     return QubitOp(symp_rep=phaseless_product._symp_csc, coeffvec=new_cfvec)


    # def multiply_by_operator(self, 
    #         operator: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> "QubitOp":
    #     """ computes right-multiplication of the internal operator 
    #     by some other operator that may contain arbitrarily many terms
    #     """
    #     operator = self.reform(operator)
    #     pauli_products = []
    #     cfvec_products = []
    #     for pauli, coeff in operator._dict().items():
    #         product = self.multiply_by_pauli({pauli:coeff})
    #         pauli_products.append(product._symp())
    #         cfvec_products.append(product.cfvec)

    #     clean_terms, clean_coeff = cleanup_symplectic(
    #             terms=np.vstack(pauli_products), 
    #             coeff=np.vstack(cfvec_products)
    #         )
    #     return QubitOp(symp_rep=clean_terms, coeffvec=clean_coeff) 
            

    # def _rotate_by_pauli(self, 
    #         pauli: Union[str, List[str], Dict[str, float], np.array, "QubitOp"]
    #     ) -> "QubitOp":
    #     """ Performs (H Omega vT \otimes v) \plus H where H is the internal operator, 
    #     Omega the symplectic form and v the symplectic representation of pauli
    #     """
    #     pauli = self.reform(pauli)
    #     assert(pauli.n_terms==1) # must be a single pauli operator

    #     commutes = self.commutes_with(pauli)
    #     rot_where_commutes = sparse.kron(commutes, pauli._symp_csc)
    #     phaseless_rotation = self._symp_csc + rot_where_commutes
    #     phaseless_rotation.data %= 2 # modulo 2
    #     phaseless_rotation = QubitOp(symp_rep = phaseless_rotation,
    #                                 coeffvec = self.cfvec)
    #     return phaseless_rotation, commutes.toarray()

    
    # def rotate_by_pauli(self, 
    #         pauli: Union[str, List[str], Dict[str, float], np.array, "QubitOp"], 
    #         angle: float = np.pi/2, 
    #         clifford_flag: bool = True
    #     ) -> "QubitOp":
    #     """ Let R(t) = e^{i t/2 Q}, then one of the following can occur:
    #     R(t) P R^\dag(t) = P when [P,Q] = 0
    #     R(t) P R^\dag(t) = cos(t) P + sin(t) (-iPQ) when {P,Q} = 0
        
    #     This operation is Clifford when t=pi/2, since cos(pi/2) P - sin(pi/2) iPQ = -iPQ.
    #     For t!=pi/2 an increase in the number of terms can be observed (non-Clifford unitary).
    #     """
    #     pauli = self.reform(pauli)
    #     phaseless_rotation, commutes = self._rotate_by_pauli(pauli)
    #     phase_mod = self.phase_modification(pauli, phaseless_rotation)
    #     # zero out phase mod where term commutes
    #     phase_mod = phase_mod*commutes
    #     commutes_inv = np.array(commutes==0, dtype=int) #inverted commutes vector
    #     if clifford_flag:
    #         # rotates by pi/2 regardless of specified angle
    #         # set clifford=False to change this behaviour
    #         phase_mod = commutes_inv - phase_mod*1j # -1j comes from rotation derivation above
    #         new_cfvec = self.cfvec*pauli.cfvec*phase_mod
    #         return QubitOp(symp_rep=phaseless_rotation._symp_csc, coeffvec=new_cfvec)
    #     else:
    #         phase_mod = commutes_inv - np.sin(angle)*phase_mod*1j
    #         new_cfvec = self.cfvec*pauli.cfvec*phase_mod
    #         # when non-Clifford the anti-commuting terms can split in two under rotation:
    #         extra_ops = self._symp_csc[np.where((commutes==1).T[0])]
    #         extra_cfv = self.cfvec[np.where((commutes==1).T[0])]*pauli.cfvec*np.cos(angle)
    #         # remove and sum coefficients of duplicate terms arising from this split
    #         clean_terms, clean_coeff = cleanup_symplectic(
    #             terms=sparse.vstack([phaseless_rotation._symp_csc, extra_ops]).toarray(), 
    #             coeff=np.vstack([new_cfvec, extra_cfv])
    #         )
    #         return QubitOp(symp_rep=clean_terms, coeffvec=clean_coeff)
            
    
    # def perform_rotations(self, 
    #         rotation_list:List[Tuple[str,float,bool]]
    #     ) -> "QubitOp":
    #     """ Allows one to perform a list of rotations sequentially,
    #     stored in the form [(pauli, angle, clifford_flag), ...]
    #     """
    #     op_copy = self.copy()
    #     for pauli, angle, clifford_flag in rotation_list:
    #         op_copy = op_copy.rotate_by_pauli(pauli,angle,clifford_flag)
    #     return op_copy


    # ###### BELOW THIS POINT SOON TO BE DEPRECATED #######
    # ###### Left for the purposes of checking      #######

    # def _dictionary_rotation(self, 
    #                         pauli_rot:str, 
    #                         angle:float=None,
    #                         clifford_flag:bool=True
    #                         )->Dict[str, float]:

    #     ## determine possible Paulis in image to avoid if statements
    #     #pauli_rot_symp_csc = pauli_to_symp_csclectic(pauli_rot)
    #     #I = np.eye(2*self.n_qubits, 2*self.n_qubits)
    #     #OmegaPtxP = np.outer(self.symform @ pauli_rot_symp_csc.T, pauli_rot_symp_csc)
    #     #P_mult_mat = (I+OmegaPtxP) % 2
    #     ## determine Pauli terms
    #     #RQRt = (self._symp_csc @ P_mult_mat) % 2
    #     #poss_ops = np.concatenate((self._symp_csc, RQRt))
    #     #poss_ops = dictionary_operator(
    #     #                                poss_ops, 
    #     #                                np.array([[0] for i in range(len(poss_ops))])
    #     #                                )
    #     def update_op(op, P, c):
    #         if P not in op:
    #             op[P] = c.real
    #         else:
    #             op[P] += c.real

    #     def commutes(P, Q):
    #         num_diff=0
    #         for Pi,Qi in zip(P,Q):
    #             if Pi=='I' or Qi=='I':
    #                 pass
    #             else:
    #                 if Pi!=Qi:
    #                     num_diff+=1
    #         return not bool(num_diff%2)

    #     op_out = {}
    #     #commuting = list(self.commuting(pauli_rot).T[0]==0)
    #     for (pauli,coeff) in self._dict().items():#,commutes in zip(self._dict.items(), commuting):
    #         if commutes(pauli, pauli_rot):#commutes:
    #             update_op(op=op_out, P=pauli, c=coeff)
    #         else:
    #             phases, paulis = zip(*[self.pauli_map[P+Q] for P,Q in zip(pauli_rot, pauli)])
    #             coeff_update = coeff*1j*np.prod(phases)
    #             if clifford_flag:
    #                 update_op(op=op_out, P=''.join(paulis), c=coeff_update)
    #             else:
    #                 update_op(op=op_out, P=pauli, c=np.cos(angle)*coeff)
    #                 update_op(op=op_out, P=''.join(paulis), c=np.sin(angle)*coeff_update)
                
    #     return op_out_op(op, P, c):
    #         if P not in op:
    #             op[P] = c.real
    #         else:
    #             op[P] += c.real

    #     def commutes(P, Q):
    #         num_diff=0
    #         for Pi,Qi in zip(P,Q):
    #             if Pi=='I' or Qi=='I':
    #                 pass
    #             else:
    #                 if Pi!=Qi:
    #                     num_diff+=1
    #         return not bool(num_diff%2)

    #     op_out = {}
    #     #commuting = list(self.commuting(pauli_rot).T[0]==0)
    #     for (pauli,coeff) in self._dict().items():#,commutes in zip(self._dict.items(), commuting):
    #         if commutes(pauli, pauli_rot):#commutes:
    #             update_op(op=op_out, P=pauli, c=coeff)
    #         else:
    #             phases, paulis = zip(*[self.pauli_map[P+Q] for P,Q in zip(pauli_rot, pauli)])
    #             coeff_update = coeff*1j*np.prod(phases)
    #             if clifford_flag:
    #                 update_op(op=op_out, P=''.join(paulis), c=coeff_update)
    #             else:
    #                 update_op(op=op_out, P=pauli, c=np.cos(angle)*coeff)
    #                 update_op(op=op_out, P=''.join(paulis), c=np.sin(angle)*coeff_update)
                
    #     return op_out


        