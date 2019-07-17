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
    atoms_ijk: np.ndarray
        The corresponding atoms of i,j,k, while the first value of the tuple will be the center atom.
    """
    distances, first_atoms, second_atoms, cell_shift_vector = AtomsDistances(positions, cell, cutoff_radius)
    Rij = []
    Rik = []
    Rjk = []
    # start index
    index = 0
    atoms_ijk = []
    # loop of center atom i 
    for i in range(len(positions)):
        # next loop when the atom is not center atom anymore
        while first_atoms[index] == i and :
            # loop of second atom
            for j in range(index, len(first_atoms)):
                # loop will be break when the first atom is not center atom anymore
                if first_atoms[j] != i:
                    break
                # loop of third atom
                for k in range(j + 1, len(first_atoms)):
                    # loop will be break when the first atom is not center atom anymore
                    if first_atoms[k] != i:
                        # then the start index need to plus one since we need to step forward to next permutation
                        index += 1
                        break
                    Rij += [distances[j]]
                    Rik += [distances[k]]
                    positions_j = positions[second_atoms[j]] + np.dot(cell_shift_vector[j], cell)
                    positions_k = positions[second_atoms[k]] + np.dot(cell_shift_vector[k], cell)
                    distances_jk_vector = (positions_j - positions_k)**2
                    distances_jk = np.sqrt(np.sum(distances_jk_vector))
                    Rjk += [distances_jk]
                    atoms_ijk += [(i, second_atoms[j], second_atoms[k])]
                    
            # if center atom is the last atom that need to be computed, the loop can stop.
            if i == len(positions) - 1:
                break
    Rij, Rik, Rjk = np.array(Rij), np.array(Rik), np.array(Rjk)
    cos_theta_ijk = (Rjk**2 - Rik**2 - Rij**2) / (-2 * Rik * Rij)
    atoms_ijk = np.array(atoms_ijk)
    
    return cos_theta_ijk, atoms_ijk