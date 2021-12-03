from pathlib import Path

import numpy as np
from openfermion import jw_configuration_state
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import jordan_wigner
from pyscf import ao2mo, gto, scf

from quantumtools.tapering import get_ground_tapered_H_using_HF_state

water_xyz_path = Path("tests/molecules/water.xyz").absolute()


def test_get_ground_tapered_H_using_HF_state():

    ### params ###
    basis = "STO-3G"
    charge = 0
    spin = 0

    ###
    full_system_mol = gto.Mole(
        atom=str(water_xyz_path),
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
    one_body_integrals = c_mat.T @ HF_scf.get_hcore() @ c_mat

    two_body_compressed = ao2mo.kernel(HF_scf.mol, c_mat)

    # get electron repulsion integrals
    eri = ao2mo.restore(1, two_body_compressed, n_orbs)  # no permutation symmetry

    # Openfermion uses physicist notation whereas pyscf uses chemists
    two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")

    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
        one_body_integrals, two_body_integrals
    )

    fermionic_molecular_hamiltonian = InteractionOperator(
        0, one_body_coefficients, 0.5 * two_body_coefficients
    )

    ### apply tapering!
    qham = jordan_wigner(fermionic_molecular_hamiltonian)
    n_qubits = 2 * full_system_mol.nao
    jw_ground_state = np.zeros((n_qubits))
    jw_ground_state[: full_system_mol.nelectron] = 1  # n occupied sites

    H_tapered = get_ground_tapered_H_using_HF_state(
        qham, jw_ground_state, check_tapering=True
    )
