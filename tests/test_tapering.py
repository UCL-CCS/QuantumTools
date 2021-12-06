
from quantumtools.tapering import get_ground_tapered_H_using_HF_state, find_all_sectors, get_rotated_operator_and_generators
from pathlib import Path
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import (
    InteractionOperator,
)
from pyscf import ao2mo, gto, scf
import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from scipy.sparse.linalg import eigsh
from functools import reduce

water_xyz_path = Path("tests/molecules/water.xyz").absolute()

# def test_get_ground_tapered_H_using_HF_state():
#
#     ### params ### 
#     basis = 'STO-3G'
#     charge = 0
#     spin = 0
#
#     ###
#     full_system_mol = gto.Mole(atom=str(water_xyz_path),
#                                basis=basis,
#                                charge=charge,
#                                spin=spin,
#                                )
#     full_system_mol.build()
#
#     HF_scf = scf.RHF(full_system_mol)
#     HF_scf.verbose = 1
#     HF_scf.conv_tol = 1e-6
#     HF_scf.kernel()
#
#     ## build H ##
#
#     c_mat = HF_scf.mo_coeff
#     n_orbs = c_mat.shape[1]
#
#     # one body terms
#     one_body_integrals = (
#             c_mat.T @ HF_scf.get_hcore() @ c_mat
#     )
#
#     two_body_compressed = ao2mo.kernel(HF_scf.mol, c_mat)
#
#     # get electron repulsion integrals
#     eri = ao2mo.restore(1, two_body_compressed, n_orbs)  # no permutation symmetry
#
#     # Openfermion uses physicist notation whereas pyscf uses chemists
#     two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")
#
#     one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
#         one_body_integrals, two_body_integrals
#     )
#
#     fermionic_molecular_hamiltonian = InteractionOperator(0, one_body_coefficients, 0.5 * two_body_coefficients)
#
#     ### apply tapering!
#     qham = jordan_wigner(fermionic_molecular_hamiltonian)
#     n_qubits = 2*full_system_mol.nao
#     jw_ground_state = np.zeros((n_qubits))
#     jw_ground_state[:full_system_mol.nelectron]=1 # n occupied sites
#
#     H_tapered_ground = get_ground_tapered_H_using_HF_state(qham, jw_ground_state, check_tapering=True)
#
#     ferm_H_sparse = get_sparse_operator(fermionic_molecular_hamiltonian)
#     if ferm_H_sparse.shape[0] < 128:
#         eigvals, eigvecs = np.linalg.eigh(ferm_H_sparse.todense())
#         min_eigval_fermionic = min(eigvals)
#     else:
#         min_eigval_fermionic = eigsh(ferm_H_sparse, k=1, which='SA')[0][0]
#     del ferm_H_sparse
#
#     H_tapered_ground_sparse = get_sparse_operator(H_tapered_ground)
#     if H_tapered_ground_sparse.shape[0] < 128:
#         eigvals, eigvecs = np.linalg.eigh(H_tapered_ground_sparse.todense())
#         min_eigval_tapered_ground = min(eigvals)
#     else:
#         min_eigval_tapered_ground = eigsh(H_tapered_ground_sparse, k=1, which='SA')[0][0]
#
#     del H_tapered_ground_sparse
#
#     print(min_eigval_tapered_ground)
#     print(min_eigval_fermionic)
#     assert np.isclose(min_eigval_tapered_ground, min_eigval_fermionic)


def test_full_spectrum_tapered():

    ### params ### 
    basis = 'STO-3G'
    charge = 0
    spin = 0

    ###
    full_system_mol = gto.Mole(atom=str(water_xyz_path),
                               basis=basis,
                               charge=charge,
                               spin=spin,
                               )
    full_system_mol.build()

    HF_scf = scf.RHF(full_system_mol)
    HF_scf.verbose = 1
    HF_scf.conv_tol = 1e-6
    HF_scf.kernel()

    ## build H ##

    c_mat = HF_scf.mo_coeff
    n_orbs = c_mat.shape[1]

    # one body terms
    one_body_integrals = (
            c_mat.T @ HF_scf.get_hcore() @ c_mat
    )

    two_body_compressed = ao2mo.kernel(HF_scf.mol, c_mat)

    # get electron repulsion integrals
    eri = ao2mo.restore(1, two_body_compressed, n_orbs)  # no permutation symmetry

    # Openfermion uses physicist notation whereas pyscf uses chemists
    two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")

    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
        one_body_integrals, two_body_integrals
    )

    fermionic_molecular_hamiltonian = InteractionOperator(0, one_body_coefficients, 0.5 * two_body_coefficients)

    ### apply tapering!
    qham = jordan_wigner(fermionic_molecular_hamiltonian)

    H_rotated, U_clifford_rotations = get_rotated_operator_and_generators(qham)

    H_full = get_sparse_operator(reduce(lambda x, y: x + y, qham))
    H_full_rotated = get_sparse_operator(reduce(lambda x, y: x + y, H_rotated))
    eigvals = eigsh(H_full, k=1, which='SA')[0][0]
    eigvals2 = eigsh(H_full_rotated, k=1, which='SA')[0][0]
    assert np.allclose(eigvals, eigvals2)

    full_spectrum_tapered = find_all_sectors(H_rotated, U_clifford_rotations)
    H_tap_ground = full_spectrum_tapered[0][1][0]
    H_tap_ground_mat = get_sparse_operator(reduce(lambda x, y: x + y, H_tap_ground))
    eigvals3 = eigsh(H_tap_ground_mat, k=1, which='SA')[0][0]
    assert np.allclose(eigvals, eigvals3)