from atomsDistances import *
import autograd.numpy as np
from atomsAngle import *
from autograd import jacobian

def cutoff(cutoff_radius, distances):
    fc = 0.5 * (np.cos((np.pi * distances) / cutoff_radius) + 1)
    return fc

def G2(positions, cell, numbers, elements, params):
    cutoff_radius = params['cutoff_radius']
    width = params['width']
    Rs = params['Rs']
    distances, first_atoms, second_atoms, offset = atomsDistances(positions, cell, cutoff_radius)
    g2 = np.exp(-width * (distances - Rs)**2 / cutoff_radius**2)
    g2 *= cutoff(cutoff_radius, distances)
    atoms_mask = np.arange(len(positions))[:, None] == first_atoms[None, :]
    
    species_mask = []
    for element in elements:
        species_mask.append(list(map(lambda x: x == element, numbers[second_atoms])) * np.ones(second_atoms.shape))
    species_mask = np.array(species_mask).T
    
    
    g2 = np.repeat(g2[None, :], len(positions), axis = 0)
    g2 = g2 * atoms_mask
    g2 = np.dot(g2, species_mask)

    return g2

# def G4(positions, cell, params):
#     cutoff_radius = params['cutoff_radius']
#     eta = params['eta']
#     zeta = params['zeta']
#     lbda = params['lbda']
#     cos, Rij, Rik, ij = newAtomsAngle(positions, cell, cutoff_radius)
#     g4 = []
#     for i in range(len(positions)):
#         mask = (ij == i)
#         g4_i = (1 + lbda * cos[mask])**zeta * np.exp(-eta * (Rij[mask]**2 + Rik[mask]**2)) 
#         g4_i = g4_i * cutoff(cutoff_radius, Rij[mask]) * cutoff(cutoff_radius, Rik[mask])
#         g4_i = 2**(1 - zeta) * np.sum(g4_i)
#         g4.append(g4_i)
#     return np.array(g4)

def G4(positions, cell, numbers, elements, params):
    cutoff_radius = params['cutoff_radius']
    eta = params['eta']
    zeta = params['zeta']
    lbda = params['lbda']
    cos, Rij, Rik, Rjk, i, jk = atomsAngle(positions, cell, cutoff_radius)
    g4 = (1 + lbda * cos)**zeta * np.exp(-eta * (Rij**2 + Rik**2 + Rjk**2) / cutoff_radius**2) 
    g4 = g4 * cutoff(cutoff_radius, Rij) * cutoff(cutoff_radius, Rik) * cutoff(cutoff_radius, Rjk)
    g4 *= 2**(1 - zeta)
    atoms_mask = np.arange(len(positions))[:, None] == i[None, :]
    
    # the shape of g4 will become (#atoms, len(g4)) and multiply atoms_mask to get the corresponding center atom's fingerprints
    g4 = np.repeat(g4[None, :], len(positions), axis = 0)
    g4 *= atoms_mask
    
    
    index = np.indices((len(elements), len(elements))).reshape(2, -1)
    mask = index[1] >= index[0]
    index = index[:, mask].T
    pairs = np.repeat(np.sort(numbers[jk])[:, None], len(index), axis = 1)
    elements = np.sort(elements)[index]

    pairs_mask = np.sum(pairs == elements, axis = 2)
    pairs_mask = np.where(pairs_mask == 2, 1, 0)

    g4 = np.dot(g4, pairs_mask)
    
    return g4