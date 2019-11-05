import autograd.numpy as np

def atomsDistances(positions, cell, cutoff_radius = 6.0, self_interaction = False):
    """ Compute the distance of every atom to its neighbors.

    
    This function computes the distances of every central atom to its neighbors. If the
    distances is larger than the cutoff radius, then the distances will be handled as 0.
    Here, periodic boundary condition is assuming true for every axis.

    Parameters:
    -----------
    positions: np.ndarray
        Atomic positions. The size of this tensor will be (N_atoms, 3), where N_atoms is the number of atoms
        in the cluster.
    cell: np.ndarray
        Periodic cell, which has the size of (3, 3)
    cutoff_radius: float
        Cutoff Radius, which is a hyper parameters. The default is 6.0 Angstrom.
    self_interaction: boolean
        Default is False, which means that results will not consider the atom itself as its neighbor.
    Returns:
    ----------
    distances: np.ndarray
        Differentialble distances array.
    first_atoms: np.ndarray
        Atoms that we observed in the cell. The np.unique of first_atoms will be np.arange of the number of
        atoms in the cell.
    second_atoms: np.ndarray
        Atoms that are considered as the neighbor atoms of first atoms. The distances of first_atoms and
        second_atoms will be computed and stored in the distances array.
    cell_shift_vector: np.ndarray
        The cell shift vector of every atom.
    """
    # Compute reciprocal lattice vectors.
    inverse_cell = np.linalg.pinv(cell).T

    # Compute distances of cell faces.
    face_dist_c = 1 / np.linalg.norm(inverse_cell, axis = 0)
    
    # We use a minimum bin size of 3 A
    bin_size = max(cutoff_radius, 3)
    
    # Compute number of bins, the minimum bin size must be [1., 1., 1.].
    nbins_c = np.maximum((face_dist_c / bin_size - (face_dist_c / bin_size) % 1), [1., 1., 1.])
    nbins = np.prod(nbins_c)
    
    # Compute the number of neighbor cell that need to be search
    neighbor_search_x, neighbor_search_y, neighbor_search_z =\
                np.ceil(bin_size * nbins_c / face_dist_c).astype(int)
    
    # Sort atoms into bins.
    scaled_positions_ic = np.dot(positions, inverse_cell) % 1
    bin_index_ic = scaled_positions_ic * nbins_c - (scaled_positions_ic * nbins_c) % 1

    # Convert Cartesian bin index to unique scalar bin index.
    bin_index_i = (bin_index_ic[:, 0] + nbins_c[0] * (bin_index_ic[:, 1] + nbins_c[1] * bin_index_ic[:, 2]))

    # atom_i contains atom index in new sort order.
    atom_i = np.argsort(bin_index_i)
    bin_index_i = bin_index_i[atom_i]

    # Compute the maximum number of atoms in a bin
    max_natoms_per_bin = np.bincount(np.int_(bin_index_i)).max()

    # Sort atoms into bins. The atoms_in_bin_ba contains the information about where the atoms located.
    atoms_in_bin_ba = -np.ones([np.int_(nbins), max_natoms_per_bin], dtype=int)

    for i in range(max_natoms_per_bin):
        # Create a mask array that identifies the first atom of each bin.
        mask = np.append([True], bin_index_i[:-1] != bin_index_i[1:])
        # Assign all first atoms.
        atoms_in_bin_ba[np.int_(bin_index_i[mask]), i] = atom_i[mask]

        # Remove atoms that we just sorted into atoms_in_bin_ba. The next
        # "first" atom will be the second and so on.
        mask = np.logical_not(mask)
        atom_i = atom_i[mask]
        bin_index_i = bin_index_i[mask]

    # Create the shift list that indicates that where the cell might shift.
    shift = []
    for x in range(-neighbor_search_x, neighbor_search_x + 1):
        for y in range(-neighbor_search_y, neighbor_search_y + 1):
            for z in range(-neighbor_search_z, neighbor_search_z + 1):
                shift += [[x, y, z]]
    
    # Therefore, the possible positions of neighborhood bin can be computed by the following code.       
    neighborbin = (bin_index_ic[:, None] + np.array(shift)[None, :]) % nbins_c
    cell_shift = ((bin_index_ic[:, None] + np.array(shift)[None, :]) - neighborbin) / nbins_c
    neighborbin = neighborbin[:, :, 0] + nbins_c[0] * (neighborbin[:, :, 1] + nbins_c[1] * neighborbin[:, :, 2])

    distances = []
    first_atoms = []
    second_atoms = []
    cell_shift_vector = []
    for i in range(len(positions)):
        # Create a mask that indicates which neighborhood bin contains atoms.
        if self_interaction:
            mask = (atoms_in_bin_ba[np.int_(neighborbin[i])] != -1)
        else:
            mask = np.logical_and(atoms_in_bin_ba[np.int_(neighborbin[i])] != -1, atoms_in_bin_ba[np.int_(neighborbin[i])] != i)
        distances_vec = positions[atoms_in_bin_ba[np.int_(neighborbin[i])]] - positions[i]
        # the distance should consider the cell shift
        distances_vec = distances_vec + np.dot(cell_shift[i], cell)[:, None]
        # make the cell shift vector for every atom instead of every bin.
        _cell_shift_vector = np.repeat(cell_shift[i][:, None], max_natoms_per_bin, axis = 1)[mask]
        distances_vec = distances_vec[mask]
        temp_distances = np.sum(distances_vec*distances_vec, axis = 1)
        temp_distances = (temp_distances)**0.5
        cutoff_mask = (temp_distances < cutoff_radius)
        _second_atoms = atoms_in_bin_ba[np.int_(neighborbin[i])][mask][cutoff_mask]
        _first_atoms = [i] * len(_second_atoms)
        _cell_shift_vector = _cell_shift_vector[cutoff_mask]
        first_atoms.extend(_first_atoms)
        second_atoms.extend(_second_atoms)
        distances.extend(temp_distances[cutoff_mask])
        cell_shift_vector.extend(_cell_shift_vector)

    distances = np.array(distances)
    cell_shift_vector = np.array(cell_shift_vector)
    first_atoms = np.array(first_atoms)
    second_atoms = np.array(second_atoms)
    
    return distances, first_atoms, second_atoms, cell_shift_vector
