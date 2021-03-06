"""Test the QHamConverter class."""

from pathlib import Path

import numpy as np
import pennylane as qml
from openfermion import QubitOperator
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp

from quantumtools.converter import QHamConverter

water_filepath = Path("tests/molecules/water.xyz").absolute()

hamiltonian = 0.5 * QubitOperator("")
hamiltonian += 0.25 * QubitOperator("X2")
hamiltonian += 0.2 * QubitOperator("Y3")

qiskit_hamiltonian = PauliSumOp.from_list(
    [("IIII", 0.5), ("IIXI", 0.25), ("IIIY", 0.2)]
)
pennylane_hamiltonian = qml.Hamiltonian(
    [0.5, 0.25, 0.2],
    [
        qml.Identity(0) @ qml.Identity(1) @ qml.Identity(2) @ qml.Identity(3),
        qml.Identity(0) @ qml.Identity(1) @ qml.PauliX(2) @ qml.Identity(3),
        qml.Identity(0) @ qml.Identity(1) @ qml.Identity(2) @ qml.PauliY(3),
    ],
)


def test_qiskit() -> None:
    converted_ham = QHamConverter(hamiltonian).convert("qiskit")
    assert type(converted_ham) is PauliSumOp
    assert np.all(converted_ham.to_matrix() - qiskit_hamiltonian.to_matrix() == 0)


def test_pennylane() -> None:
    converted_ham = QHamConverter(hamiltonian).convert("pennylane")
    assert type(converted_ham) is qml.Hamiltonian
    assert pennylane_hamiltonian.compare(pennylane_hamiltonian)


if __name__ == "__main__":
    pass
