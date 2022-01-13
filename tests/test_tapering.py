from quantumtools.tapering import (find_all_sectors,
                                   get_rotated_operator_and_generators,
                                   get_ground_tapered_H_analytic,
                                   get_ground_tapered_H_by_qubit_state)
from pathlib import Path
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import (
    InteractionOperator,
)
from openfermion import count_qubits
from pyscf import ao2mo, gto, scf
import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from scipy.sparse.linalg import eigsh

water_xyz_path = Path("tests/molecules/water.xyz").absolute()
H2_xyz_path = Path("tests/molecules/H2.xyz").absolute()

def get_Hamiltonian(xyz_path):
    ### params ### 
    basis = 'STO-3G'
    charge = 0
    spin = 0

    ###
    full_system_mol = gto.Mole(atom=str(xyz_path),
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

    return qham

H_test_water = get_Hamiltonian(water_xyz_path)
H_test_hydro = get_Hamiltonian(H2_xyz_path)

def test_find_all_sectors(H_test_hydro):

    H_rotated, sym_generators_list = get_rotated_operator_and_generators(H_test_hydro)
    full_spectrum_tapered = find_all_sectors(H_rotated, sym_generators_list)
    ordered_eigs = np.array([res[0] for res in full_spectrum_tapered])

    H_full = get_sparse_operator(H_test_hydro)
    eigvals, eigvecs = np.linalg.eigh(H_full.todense())
    assert np.allclose(ordered_eigs, eigvals[:len(ordered_eigs)])


def test_get_ground_tapered_H_analytic(H_test_water):

    H_ground_tap = get_ground_tapered_H_analytic(H_test_water,
                                                 check_tapering=False)
    gs_energy_tap = eigsh(get_sparse_operator(H_ground_tap), k=1, which='SA')[0][0]
    gs_energy_full = eigsh(get_sparse_operator(H_test_water), k=1, which='SA')[0][0]
    assert np.allclose(gs_energy_tap, gs_energy_full)


# def test_get_ground_tapered_H_by_qubit_state(H_test_water):
#     nelectron=10
#     n_qubits= count_qubits(H_test_water)
#     JW_state = np.zeros(n_qubits)
#     JW_state[:nelectron] = np.ones(nelectron)
#
#     H_ground_tap = get_ground_tapered_H_by_qubit_state(H_test_water,
#                                                        JW_state,
#                                                        check_tapering=False)
#
#     gs_energy_tap = eigsh(get_sparse_operator(H_ground_tap), k=1, which='SA')[0][0]
#     gs_energy_full = eigsh(get_sparse_operator(H_test_water), k=1, which='SA')[0][0]
#     assert np.allclose(gs_energy_tap, gs_energy_full)

