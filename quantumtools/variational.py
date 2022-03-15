from functools import cached_property
from symred.symplectic_form import PauliwordOp
from functools import cached_property
import pyzx as zx
from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.circuit import ParameterVector
from qiskit.opflow import CircuitStateFn
import numpy as np
from collections import Counter
from operator import add
from functools import reduce
from copy import deepcopy
from scipy.optimize import minimize

class Ansatz:
    """ Class for building and evaluating expectation values of paramtrized Ansatze
    """
    def __init__(self, 
        ansatz_operator: PauliwordOp, 
        ref_state: np.array = None
        ) -> None:
        assert(set(ref_state).issubset([0,1])), 'Reference state is note defined over binary integers 0 and 1'
        assert(ansatz_operator.n_qubits == len(ref_state)), 'reference state contains incorrect number of qubits'
        self.ansatz_operator = ansatz_operator
        self.ref_state = ref_state
        self.trotter_number = 2

    @cached_property
    def to_QuantumCircuit(self):
        """ Convert the Ansatz to a first-order Trotterized quantum circuit 
        """
        def qiskit_ordering(indices):
            """ we index from left to right - in Qiskit this ordering is reversed
            """
            return self.ansatz_operator.n_qubits - 1 - indices

        qc = QuantumCircuit(self.ansatz_operator.n_qubits)
        for i in qiskit_ordering(np.where(self.ref_state==1)[0]):
            qc.x(i)

        circuit_instructions = {}
        for step, (X,Z) in enumerate(zip(self.ansatz_operator.X_block, self.ansatz_operator.Z_block)):
            # locations for H and S gates to transform into Pauli Z basis
            H_indices = qiskit_ordering(np.where(X)[0])[::-1]
            S_indices = qiskit_ordering(np.where(X & Z)[0])[::-1]
            # CNOT cascade indices
            CNOT_indices = qiskit_ordering(np.where(X | Z)[0])[::-1]

            circuit_instructions[step] = {'H_indices':H_indices, 
                                        'S_indices':S_indices, 
                                        'CNOT_indices':CNOT_indices,
                                        'RZ_index':CNOT_indices[-1]}

        def CNOT_cascade(cascade_indices, reverse=False):
            index_pairs = list(zip(cascade_indices[:-1], cascade_indices[1:]))
            if reverse:
                index_pairs = index_pairs[::-1]
            for source, target in index_pairs:
                qc.cx(source, target)

        def circuit_from_step(angle, H_indices, S_indices, CNOT_indices, RZ_index):
            # to Pauli X basis
            for i in S_indices:
                qc.sdg(i)
            # to Pauli Z basis
            for i in H_indices:
                qc.h(i)
            # compute parity
            CNOT_cascade(CNOT_indices)
            qc.rz(-2*angle, RZ_index)
            CNOT_cascade(CNOT_indices, reverse=True)
            for i in H_indices:
                qc.h(i)
            for i in S_indices:
                qc.s(i)

        def circ_from_instructions(instructions, angles, trotter_number=self.trotter_number):
            angles= np.array(angles)/trotter_number
            assert(len(angles)==len(instructions))
            for trot_step in range(trotter_number):
                for step, gate_indices in instructions.items():
                    qc.barrier()
                    circuit_from_step(angles[step], *gate_indices.values())
            return qc
        
        return circ_from_instructions(circuit_instructions, ParameterVector('P', self.ansatz_operator.n_terms))

    def parametrize_circuit(self, params: np.array):
        """ Bind fixed parameter values to the Ansatz circuit for evaluation
        """
        qc = self.to_QuantumCircuit.copy()
        variables   = qc.parameters
        assert(len(params)==len(variables))
        bind_params = {p:v for p,v in zip(variables, params)}
        bound_qc    = qc.bind_parameters(bind_params)

        return bound_qc

    def ZX_calculus_reduction(self, qc: QuantumCircuit):
        """ Simplify the circuit via ZX calculus using PyZX... 
        Only works on parametrized circuits!
        """
        # to perform ZX-calculus optimization
        qc_qasm = qc.qasm()
        qc_pyzx = zx.Circuit.from_qasm(qc_qasm)
        g = qc_pyzx.to_graph()
        zx.full_reduce(g) # simplifies the Graph in-place
        g.normalize()
        c_opt = zx.extract_circuit(g.copy())
        simplified_qc = QuantumCircuit.from_qasm_str(c_opt.to_qasm())

        return simplified_qc

class VariationalAlgorithm(Ansatz):
    """ class for executing VQE workloads
    """
    backend = BasicAer.get_backend('qasm_simulator')

    def __init__(self, 
        observable_operator: PauliwordOp, 
        ansatz_operator: PauliwordOp, 
        ref_state: np.array
        ) -> None:
        """
        """
        assert(observable_operator.n_qubits == ansatz_operator.n_qubits)
        super().__init__(ansatz_operator, ref_state)
        self.observable_operator = observable_operator
    
    @cached_property
    def QWC_grouping(self):
        """ group Hamiltonian terms that may be estimated simultaneously
        """
        term_grouping = {}
        for term in self.observable_operator.to_dictionary:
            X = [i for i,Pi in enumerate(term) if Pi=='X']
            Y = [i for i,Pi in enumerate(term) if Pi=='Y']
            key = f'{X}_{Y}'
            if key not in term_grouping:
                term_grouping[key] = {'paulis':[term]}
            else:
                term_grouping[key]['paulis'].append(term)
                
        return term_grouping

    def measure_circuit(self, qc: QuantumCircuit, n_shots: int):
        """
        """
        qc.measure_all()
        # if n_shots exceeds Qiskit's limit, separate into multiple jobs:
        if n_shots > 2**15:
            jobs = [(2**15) for i in range(n_shots//(2**15))]
            extra = n_shots%(2**15)
            if extra != 0:
                jobs.append(extra)
        else:
            jobs = [n_shots]
        input=[]
        for shots in jobs:
            job = execute(qc, self.backend, shots=shots)
            counts = job.result().get_counts()
            input.append(counts)
        counts = dict(reduce(add, (Counter(dict(x)) for x in input)))

        return counts

    def _noisy_expectation(self, grouped_measurements, n_shots):
        """ evaluate expectation value from relative frequency of measurement outcomes per group
        """
        eigvals={'I':{'0':1, '1':1}, 'Z':{'0':1, '1':-1}}
        def Z_expectation(pauli, bitstring):
            out=1
            for Pi,bi in zip(pauli, bitstring):
                out*=eigvals[Pi][bi]
            return out

        expectation=0
        for XY, countdata in grouped_measurements.items():
            paulis = countdata['paulis']
            counts = countdata['counts']
            for P in paulis:
                P_copy = list(deepcopy(P))
                for i,Pi in enumerate(P_copy):
                    if Pi in ['X', 'Y']:
                        P_copy[i]='Z'
                P_copy = ''.join(P_copy)
                relative_weight=sum([Z_expectation(P_copy, state)*freq/n_shots for state, freq in counts.items()])
                expectation += self.observable_operator.to_dictionary[P]*relative_weight
                
        return expectation.real

    def noisy_expectation(self, x: np.array, n_shots: int = 2**10):
        bound_qc = self.parametrize_circuit(x)
        QWC_group_measurements = deepcopy(self.QWC_grouping)
        for XY in self.QWC_grouping:
            qc_copy = bound_qc.copy()
            X_Y = XY.split('_')
            X = [int(i) for i in X_Y[0][1:-1].split(',') if i!='']
            Y = [int(i) for i in X_Y[1][1:-1].split(',') if i!='']
            # tranform onto correct basis for current QWC group
            for i in X:
                q_pos = self.ansatz_operator.n_qubits-1-i
                qc_copy.h(q_pos)
            for i in Y:
                q_pos = self.ansatz_operator.n_qubits-1-i
                qc_copy.sdg(q_pos)
                qc_copy.h(q_pos)
            QWC_group_measurements[XY]['counts'] = self.measure_circuit(qc_copy, n_shots)
        return self._noisy_expectation(QWC_group_measurements, n_shots)

    def observable_estimation(self, x: np.array, n_shots=2**10, exact=False):
        """
        """
        if exact:
            bound_qc = self.parametrize_circuit(x)
            psi = CircuitStateFn(bound_qc).to_spmatrix()
            energy = psi @ self.observable_operator.to_sparse_matrix @ psi.transpose()
            return energy[0,0].real
        else:
            return self.noisy_expectation(x, n_shots)

    def gradient(self, x, n_shots=2**10, exact=False):
        """ return the gradient at x w.r.t. to the observable operator
        """
        gradient_vec = []
        for i in range(self.to_QuantumCircuit.num_parameters):
            diff_vec = np.zeros(self.to_QuantumCircuit.num_parameters)
            diff_vec[i] = np.pi/4
            gradient_vec.append(self.observable_estimation(x+diff_vec, n_shots, exact) 
                                - self.observable_estimation(x-diff_vec, n_shots, exact))
        return np.array(gradient_vec)
            
    def VQE(self, 
        init_params=None, 
        optimizer='SLSQP', 
        maxiter=10, 
        gradient_descent=True, 
        n_shots=2**10, 
        n_realize=1, 
        exact=False
        ):
        """
        """
        interim_values = {'values':[], 'params':[], 'gradients':[]}

        def objective(x):
            samples = np.array([self.observable_estimation(x, n_shots, exact) for itr in range(n_realize)])
            energy = np.mean(samples)
            interim_values['params'].append(x)
            interim_values['values'].append(energy)
            return energy

        def derivative(x):
            grad = self.gradient(x, n_shots, exact)
            interim_values['gradients'].append(grad)
            return grad

        if init_params is None:
            init_params = np.zeros(self.to_QuantumCircuit.num_parameters)

        vqe_result = minimize(
            objective, 
            x0=init_params, 
            jac=derivative,
            method=optimizer,
            options={'maxiter':maxiter}
        )

        return vqe_result, interim_values




