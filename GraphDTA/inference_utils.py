# GraphDTA/inference_utils.py
from rdkit import Chem
import numpy as np

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    atoms = mol.GetAtoms()
    atom_features = [[atom.GetAtomicNum()] for atom in atoms]

    edge_index = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])

    return len(atoms), atom_features, edge_index

def seq_cat(seq):
    amino_dict = {aa: idx + 1 for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    return [amino_dict.get(aa, 0) for aa in seq]
