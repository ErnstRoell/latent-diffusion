"""
https://appaeco.wordpress.com/2024/11/13/displaying-molecular-graph-using-2d-coordinates-with-networkx/
"""

# standard
import os as os
import copy as copy
import random as random

# plotting
import matplotlib.pyplot as plt

# rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


# mol with atom index
def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def getMoleculeGraph(mol):
    mol = mol_with_atom_index(mol)
    moleculegraph = {
        "mol": mol,
        "smiles": Chem.MolToSmiles(mol),
        "atoms": [a.GetIdx() for a in mol.GetAtoms()],
        "types": [a.GetAtomicNum() for a in mol.GetAtoms()],
        "bonds": [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()],
    }
    # display(mol)
    return moleculegraph


# display molecule as img of graph and reduced graph
def compute_pos(moleculegraph):
    mol = moleculegraph["mol"]

    # formatting mol
    mol = mol_with_atom_index(mol)
    AllChem.Compute2DCoords(mol)
    # compute position
    mol_con = [c.GetPositions()[:, :2] for c in mol.GetConformers()][0]
    return mol_con


# |%%--%%| <z3D2bN8QHG|81o0coMMdh>

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
mol = Chem.MolFromSmiles(smiles)
mol = mol_with_atom_index(mol)
moleculegraph = getMoleculeGraph(mol)

print(moleculegraph)

pos = compute_pos(moleculegraph)

plt.scatter(pos[:, 0], pos[:, 1])
# |%%--%%| <81o0coMMdh|VPJEnVwd3q>

dataset = QM9(root="./data")
