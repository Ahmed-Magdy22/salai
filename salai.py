# app.py
import streamlit as st
import torch
import pandas as pd
import numpy as np
import os
import time
import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, PPBuilder
import py3Dmol

from GraphDTA.models.ginconv import GINConvNet
from GraphDTA.inference_utils import smile_to_graph, seq_cat
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# Setup
ROOT_DIR = Path(__file__).parent
MODEL_PATH = ROOT_DIR / "model_GINConvNet_combined.model"
KIBA_DATA_PATH = ROOT_DIR / "data" / "kiba_test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GINConvNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

st.set_page_config(page_title="Protein-Ligand Interaction Prediction", layout="centered")
st.title("🤜 Predict Protein-Ligand Interaction (GraphDTA)")

uploaded_protein = st.file_uploader("🔬 Upload a PDB file", type=["pdb"])
predict_button = st.button("🔍 Predict Top 5 Ligands")

def extract_sequence_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)
    ppb = PPBuilder()
    sequence = ""
    for pp in ppb.build_peptides(structure):
        sequence += str(pp.get_sequence())
    return sequence

if predict_button and uploaded_protein:
    start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        protein_pdb = tmp_path / "protein.pdb"
        protein_pdb.write_bytes(uploaded_protein.getvalue())
        sequence = extract_sequence_from_pdb(str(protein_pdb))

        if not sequence:
            st.error("Could not extract protein sequence from PDB.")
            st.stop()

        protein_tensor = torch.tensor(seq_cat(sequence), dtype=torch.long).unsqueeze(0).to(device)

        df = pd.read_csv(KIBA_DATA_PATH)
        smiles_col = next((col for col in df.columns if 'smiles' in col.lower()), None)
        if not smiles_col:
            st.error("SMILES column not found.")
            st.stop()

        smiles_list = df[smiles_col].tolist()[:1000]
        data_list, smiles_valid = [], []

        for smile in smiles_list:
            try:
                c_size, features, edge_index = smile_to_graph(smile)
                x = torch.tensor(features, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                batch = torch.zeros(x.size(0), dtype=torch.long)
                data = Data(x=x, edge_index=edge_index, target=protein_tensor, batch=batch)
                data_list.append(data)
                smiles_valid.append(smile)
            except:
                continue

        if not data_list:
            st.error("No valid ligand graphs.")
            st.stop()

        loader = DataLoader(data_list, batch_size=64)
        predictions = []
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                output = model(batch)
                predictions.extend(output.cpu().numpy())

        results = sorted(zip(smiles_valid, predictions), key=lambda x: x[1], reverse=True)[:5]

        st.subheader("🔝 Top 5 Predicted Ligands")
        st.dataframe(pd.DataFrame(results, columns=["SMILES", "Score"]))

        for i, (smile, score) in enumerate(results, 1):
            mol = Chem.MolFromSmiles(smile)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            block = Chem.MolToPDBBlock(mol)

            st.markdown(f"### 🧪 Ligand #{i} — Score: {float(score):.4f}")
            viewer = py3Dmol.view(width=600, height=400)
            viewer.addModel(block, 'pdb')
            viewer.setStyle({'stick': {}})
            viewer.setBackgroundColor('white')
            viewer.zoomTo()
            st.components.v1.html(viewer._make_html(), height=420)

        st.success(f"✅ Completed in {time.time() - start:.2f} seconds.")
