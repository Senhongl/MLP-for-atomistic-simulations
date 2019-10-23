from atomsDistances import *
import autograd.numpy as np

def atomsAngle(positions, cell, cutoff_radius):
    """ Compute the cosine value of angle of every atom to its neighbors.
    
    This function will compute the cosine value of angle of three atoms according 
    to the distances among these atoms.
    
    Parameters:
    -----------
    positions: np.ndarray
        Atomic positions. The size of this tensor will be (N_atoms, 3), where N_atoms is the number of atoms
        in the cluster.
    cell: np.ndarray
        Periodic cell, which has the size of (3, 3)
    cutoff_radius: float
        Cutoff Radius, which is a hyper parameters. The default is 6.0 Angstrom.
    Returns:
    -----------
    cos_theta_ijk: np.ndarray
        The cosine value of angles of atoms i,j,k, while the center atom is i. The cosine value is computed by
        the following fomula, cos_ijk = (Rjk^2 - Rik^2 - Rij^2) / (-2 * Rik * Rij).
    Rij: np.ndarray
        The distances between i and j
    Rik: np.ndarray
        The distances between i and k
    """
    distances, first_atoms, second_atoms, cell_shift_vector = atomsDistances(positions, cell, cutoff_radius)
    cos_theta_ijk = []
    Rij = []
    Rik = []
    Rjk = []
    center_atoms = []
    jk = []

    for i in range(len(positions)):
        # first find the center atoms
        center_atom_mask = (first_atoms == i)
        i_temp = first_atoms[center_atom_mask]
        # then the second and third atoms
        j_temp = second_atoms[center_atom_mask]

        index = np.indices((len(i_temp), len(i_temp))).reshape(2, -1)
        mask = (index[1] > index[0])
        # make a combination mask
        index = index[:, mask]
        
        _Rij = distances[center_atom_mask][index[0]]
        _Rik = distances[center_atom_mask][index[1]]
        positions_j = (positions[j_temp] + np.dot(cell_shift_vector[center_atom_mask], cell))[index[0]]
        positions_k = (positions[j_temp] + np.dot(cell_shift_vector[center_atom_mask], cell))[index[1]]
        distances_jk_vector = (positions_j - positions_k)**2
        _Rjk = (np.sum(distances_jk_vector, axis = 1))**0.5
        cos_theta_ijk.extend((_Rjk**2 - _Rik**2 - _Rij**2) / (-2 * _Rik * _Rij))
        center_atoms.extend(i_temp[index[0]])
        Rij.extend(_Rij)
        Rik.extend(_Rik)
        Rjk.extend(_Rjk)
        jk.extend(np.concatenate((j_temp[index[0]][:, None], j_temp[index[1]][:, None]), axis = 1))

    cos_theta_ijk = np.array(cos_theta_ijk)
    Rij = np.array(Rij)
    Rik = np.array(Rik)
    Rjk = np.array(Rjk)
    center_atoms = np.array(center_atoms)
    jk = np.array(jk)

    return cos_theta_ijk, Rij, Rik, Rjk, center_atoms, jk