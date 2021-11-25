# QuantumTools
Random useful quantum computing for chemistry things.

## Install
From the root of the project:
```
pip install poetry
poetry install
```

## Add a dependency
```
poetry add <dependency>
```

optionally add the `-D` flag if the dependency is a development dependency (e.g. testing, linting or documentation tools.)

## QHamConverter

Is a class which takes in either an openfermion.QubitOperator or dict representing a qubit hamiltonian.
It creates an intermediate representation of the hamiltonian, e.g.:

```
hamiltonian = QubitOperator('X3',1), + QubitOperator('X1 Y2 Z3', 0.5) + QubitOperator('X0 I2 Y3 Z4', 0.5)

qhc = QHamConverter(hamiltonian)

qhc._intermediate = {"IIIX": 1, "IXYZ":0.5, "XIXY":0.5}`
```

is a 4 qubit hamiltonian constructed of Pauli terms. Only I,X,Y,Z are allowed, each qubit must have a Pauli or Identity associated with it, and numbering is zero indexed from the left.