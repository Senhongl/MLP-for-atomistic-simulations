import autograd.numpy as np

def AtomsDistances(positions, cell cutoff_radius = 6.0):
    """ Compute the distance of every atom to its neighbors.

    This function utilizes the pytorch to compute the differentiable distances tensor,
    which represents the distances of every central atom to its neighbors. If the
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
    Return:
    ----------
    distances: np.ndarray
    Differentialble distances array.
    """
    # Compute reciprocal lattice vectors.
    inverse_cell = np.linalg.pinv(cell).T

    # Compute distances of cell faces.
    face_dist_c = 1 / np.linalg.norm(inverse_cell, axis = 0)

    # Compute number of bins
    nbins_c = np.maximum((face_dist_c / cutoff_radius - (face_dist_c / cutoff_radius) % 1), [1., 1., 1.])
    nbins = np.prod(nbins_c)

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

    # Make sure that all atoms have been sorted into bins.
    assert len(atom_i) == 0
    assert len(bin_index_i) == 0

    # Create the shift list that indicates that where the cell might shift.
    shift = []
    for neighbor_search_x in range(-1, 2):
        for neighbor_search_y in range(-1, 2):
            for neighbor_search_z in range(-1, 2):
                shift += [[neighbor_search_x, neighbor_search_y, neighbor_search_z]]
    
    # Therefore, the possible positions of neighborhood bin can be computed by the following code.       
    neighborbin = (bin_index_ic[:, None] + np.array(shift)[None, :]) % nbins_c
    cell_shift = ((bin_index_ic[:, None] + np.array(shift)[None, :]) - neighborbin) / nbins_c
    neighborbin = neighborbin[:, :, 0] + nbins_c[0] * (neighborbin[:, :, 1] + nbins_c[1] * neighborbin[:, :, 2])

    distances = []
    for i in range(len(positions)):
        # Create a mask that indicates which neighborhood bin contains atoms.
        mask = np.logical_and(atoms_in_bin_ba[np.int_(neighborbin[i])] != -1, atoms_in_bin_ba[np.int_(neighborbin[i])] != i)

        distances_vec = positions[atoms_in_bin_ba[np.int_(neighborbin[i])]] - positions[i]
        # the distance should consider the cell shift
        distances_vec = distances_vec + np.dot(cell_shift[i], cell)[:, None]
        distances_vec = distances_vec[mask]
        temp_distances = np.sum(distances_vec*distances_vec, axis = 1)
        temp_distances = (temp_distances)**0.5
        cutoff_mask = (temp_distances < cutoff_radius)
        distances.extend(temp_distances[cutoff_mask])

    distances = np.array(distances)
    return distances