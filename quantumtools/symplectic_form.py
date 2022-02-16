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


class Pauliword():

    def __init__(self, pauli_string, coeff):
        self.pauli_string = pauli_string
        self._valid_pauliword()

        self.n_qubits = len(pauli_string)
        self.phase = pauli_string
        self.coeff = coeff

        self._init_symplectic()

    def __str__(self):
        return f'{self.coeff} {self.pauli_string}'

    def _valid_pauliword(self):
        """
        check to see if Pauli is made up of single qubit Pauli operators only
        """
        paulis = set(['X', 'Z', 'Y', 'I'])
        assert (set(self.pauli_string).issubset(paulis)), 'pauliword must only contain X,Y,Z,I terms'

    def _init_symplectic(self):
        symp = np.zeros(2 * self.n_qubits, dtype=int)

        char_aray = np.array(list(self.pauli_string), dtype=str)
        X_loc = (char_aray == 'X')
        Z_loc = (char_aray == 'Z')
        Y_loc = (char_aray == 'Y')

        symp[:self.n_qubits] += X_loc
        symp[self.n_qubits:] += Z_loc
        symp[:self.n_qubits] += Y_loc
        symp[self.n_qubits:] += Y_loc

        self.vec = symp

        self.X_block = symp[:self.n_qubits]
        self.Z_block = symp[self.n_qubits:]
        self.Y_count = sum(Y_loc)

    def commutes(self, Pword) -> bool:
        # checks qubit number is same
        assert (self.n_qubits == Pword.n_qubits), 'Pauliwords defined for different number of qubits'

        gf2_comm = (self.Z_block @ Pword.X_block + self.X_block @ Pword.Z_block) % 2
        return not bool(gf2_comm)

    def signless_multiplication_symp(self, Pword) -> np.array:
        pauli_mult_symp_vec = (self.vec + Pword.vec) % 2
        return pauli_mult_symp_vec

    def signless_multiplication_pword(self, Pword):
        pauli_mult_symp_vec = self.signless_multiplication_symp(Pword)
        new_p_string = symplectic_to_string(pauli_mult_symp_vec)
        return Pauliword(new_p_string, 1)

    def __mul__(self, Pword):
        signless_prod_Pword = self.signless_multiplication_pword(Pword)
        num_sign_flips = (self.X_block @ Pword.Z_block)
        sign_change = (-1) ** num_sign_flips

        # mapping from sigma to tau representation
        full_Y_count = self.Y_count + Pword.Y_count
        sigma_tau_compensation = (-1j) ** full_Y_count

        # back from tau to sigma (note uses output Pword)
        tau_sigma_compensation = (1j) ** signless_prod_Pword.Y_count

        phase_mod = sign_change * sigma_tau_compensation * tau_sigma_compensation

        overall_phase = phase_mod * self.coeff * Pword.coeff

        P_out = Pauliword(signless_prod_Pword.pauli_string, overall_phase)
        return P_out

    def __add__(self):
        # TODO: function for addition!
        pass
    def __sub__(self):
        # TODO: function for subtraction!
        pass
    def __eq__(self, Pword):
        """
        checks equivlance of current Pauliword with different Pauliword
        aka defines the  == operation
        """
        if (self.pauli_string, self.coeff) == (Pword.pauli_string, Pword.coeff):
            return True
        else:
            return False


class linear_combination_of_pauliwords():

    def __init__(self, list_pauliwords, n_qubits):
        self.list_pauliwords = list_pauliwords
        self.n_qubits = n_qubits

    def _init_symplectic_matrix(self):

        self.symp_matrix = np.zeros((len(self.list_pauliwords), 2*self.n_qubits), dtype=int)
        self.coeff_vec = np.zeros((len(self.list_pauliwords), 1), dtype=complex)
        self._comm_matrix = np.zeros((len(self.list_pauliwords), 2*self.n_qubits), dtype=int)
        self._sign_matrix = np.zeros((len(self.list_pauliwords), 2 * self.n_qubits), dtype=int)
        for ind, Pword in enumerate(self.list_pauliwords):
            self.symp_matrix[ind] = Pword.vec
            self.coeff_vec[ind] = Pword.coeff

            self._comm_matrix = np.hstack((Pword.Z_block, Pword.X_block))
            self._sign_matrix[ind, :self.n_qubits] = Pword.X_block
        return None


    def commutes(self, alt_linear_combination_pauliwords):
        pass

