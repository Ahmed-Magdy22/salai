from rdkit import Chem
import numpy as np

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    # Based on GraphDTA official repo but updated to fix deprecation warnings
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),
                              ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                               'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                               'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                               'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                               'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5]) +  # Updated to use GetTotalValence() instead of deprecated GetValence()
        [atom.GetIsAromatic()]
    )

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None, None, None  # Return None values to indicate invalid SMILES

    # Check for valid structure before proceeding
    try:
        Chem.SanitizeMol(mol)
        atoms = mol.GetAtoms()
        features = [atom_features(atom) for atom in atoms]
        # Pad features to length 78
        padded_features = [np.pad(f, (0, 78 - len(f)), 'constant') if len(f) < 78 else f[:78] for f in features]

        edge_index = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edge_index.append([start, end])
            edge_index.append([end, start])

        return len(atoms), padded_features, edge_index
    except Exception:
        return None, None, None  # Return None if sanitization fails

def seq_cat(seq):
    amino_dict = {aa: idx + 1 for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    return [amino_dict.get(aa, 0) for aa in seq]