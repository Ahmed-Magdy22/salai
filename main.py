import streamlit as st
import torch
import pandas as pd
import os
import time
import tempfile
from pathlib import Path
from rdkit import Chem
# Import AllChem conditionally to avoid drawing-related errors
try:
    from rdkit.Chem import AllChem
    HAS_RDKIT_DRAW = True
except ImportError:
    HAS_RDKIT_DRAW = False
    st.warning("RDKit drawing functionality is not available. Some visualization features may be limited.")
from Bio.PDB import PDBParser, PPBuilder
import py3Dmol
import streamlit.components.v1 as components
import numpy as np

from GraphDTA.models.ginconv import GINConvNet
from GraphDTA.inference_utils import smile_to_graph, seq_cat
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Import the chemical database integration with fallback options
try:
    from chemical_databases import search_pubchem, search_chembl, get_drugbank_approved_drugs
except ImportError as e:
    st.error(f"Error importing chemical_databases: {str(e)}")
    # Define fallback functions
    def get_drugbank_approved_drugs(max_compounds=200):
        return pd.DataFrame([
            {"name": "Aspirin", "compound_iso_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "source": "Default"},
            {"name": "Ibuprofen", "compound_iso_smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "source": "Default"},
        ])

    def search_pubchem(query, max_compounds=100):
        st.warning("PubChem search unavailable")
        return get_drugbank_approved_drugs()

    def search_chembl(query, max_compounds=100):
        st.warning("ChEMBL search unavailable")
        return get_drugbank_approved_drugs()

# Import the protein interaction module
from protein_interaction import ProteinInteractionPredictor, download_pdb, create_protein_complex, calculate_docking_score

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

# Define helper functions
def extract_sequence_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)
    ppb = PPBuilder()
    sequence = ""
    for pp in ppb.build_peptides(structure):
        sequence += str(pp.get_sequence())
    return sequence

st.set_page_config(page_title="Protein Interaction Prediction", layout="centered")

# Create tabs for different analysis modes
tab1, tab2 = st.tabs(["🧪 Protein-Ligand Interaction", "🔄 Protein-Protein Interaction"])

with tab1:
    st.title("🤜 Predict Protein-Ligand Interaction (GraphDTA)")

    # Add data source selection
    st.sidebar.title("Data Source Settings")
    data_source = st.sidebar.selectbox(
        "Choose ligand data source:",
        ["DrugBank (Approved Drugs)", "PubChem", "ChEMBL", "Local CSV (data/kiba_test.csv)"],
        index=0
    )

    if data_source in ["PubChem", "ChEMBL"]:
        search_query = st.sidebar.text_input("Search term (e.g., drug name, SMILES, disease)", "aspirin")
        max_results = st.sidebar.slider("Maximum results to fetch", 10, 200, 50)
        search_button = st.sidebar.button("Search Database")

    # Add visualization settings
    st.sidebar.title("Visualization Settings")
    viz_style = st.sidebar.selectbox(
        "Protein visualization style:",
        ["Cartoon", "Stick", "Line", "Cross", "Sphere"],
        index=0
    )
    color_scheme = st.sidebar.selectbox(
        "Color scheme:",
        ["Chain", "Spectrum", "Secondary Structure", "Residue Type", "B-factor"],
        index=0
    )
    spin_model = st.sidebar.checkbox("Spin model", value=True)
    background_color = st.sidebar.selectbox(
        "Background color:",
        ["White", "Black", "Gray", "Light Blue"]
    )

    uploaded_protein = st.file_uploader("🔬 Upload a PDB file", type=["pdb"], key="ligand_tab")

    # Display 3D preview when protein is uploaded
    if uploaded_protein is not None:
        st.subheader("🔍 Protein Structure Preview")

        # Create columns for the preview and info
        col1, col2 = st.columns([2, 1])

        # Process uploaded protein file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
            tmp.write(uploaded_protein.getvalue())
            temp_pdb_path = tmp.name

        with open(temp_pdb_path, 'r') as f:
            protein_data = f.read()

        # Extract and display protein info
        with col2:
            st.markdown("### Protein Information")
            try:
                # Extract sequence
                sequence = extract_sequence_from_pdb(temp_pdb_path)
                sequence_display = sequence[:50] + "..." if len(sequence) > 50 else sequence

                # Get metadata from PDB file
                pdb_id = "Unknown"
                title = "Unknown"
                for line in protein_data.splitlines():
                    if line.startswith("HEADER"):
                        parts = line.split()
                        if len(parts) > 1:
                            pdb_id = str(parts[-1])
                    if line.startswith("TITLE"):
                        title = line[10:].strip()

                # Display metadata
                st.markdown(f"**PDB ID:** {pdb_id}")
                st.markdown(f"**Title:** {title}")
                st.markdown(f"**Sequence Length:** {len(sequence)} amino acids")
                st.markdown(f"**Sequence:** `{sequence_display}`")

                # Add download button for the PDB
                st.download_button(
                    label="Download PDB File",
                    data=protein_data,
                    file_name="protein.pdb",
                    mime="chemical/x-pdb",
                    key="download_pdb_file_main"
                )
            except Exception as e:
                st.error(f"Error processing protein file: {str(e)}")

        # Display 3D preview
        with col1:
            # Map user selections to py3Dmol parameters
            style_map = {
                "Cartoon": "cartoon",
                "Stick": "stick",
                "Line": "line",
                "Cross": "cross",
                "Sphere": "sphere"
            }

            color_map = {
                "Chain": "chain",
                "Spectrum": "spectrum",
                "Secondary Structure": "sstruc",
                "Residue Type": "restype",
                "B-factor": "b"
            }

            bg_color_map = {
                "White": "white",
                "Black": "black",
                "Gray": "gray",
                "Light Blue": "lightblue"
            }

            # Create 3D viewer
            viewer = py3Dmol.view(width=600, height=400)
            viewer.addModel(protein_data, 'pdb')

            # Apply selected style
            viewer.setStyle({}, {style_map[viz_style]: {'colorscheme': 'chain'}})

            # Apply other visualization settings
            viewer.setBackgroundColor(bg_color_map[background_color])
            viewer.zoomTo()
            if spin_model:
                viewer.spin(True)

            # Add surface representation with transparency
            if st.sidebar.checkbox("Show surface", value=False):
                transparency = st.sidebar.slider("Surface transparency", 0.0, 1.0, 0.7)
                viewer.addSurface(py3Dmol.VDW, {'opacity': 1.0 - transparency, 'color':'white'})

            # Render the viewer
            components.html(viewer._make_html(), height=420)

            # Help text
            st.caption("👆 Tip: Click and drag to rotate. Scroll to zoom. Shift+click to translate.")

    predict_ligand_button = st.button("🔍 Predict Top 5 Ligands")

    # Use session state to persist results
    if 'ligand_results' not in st.session_state:
        st.session_state['ligand_results'] = None
        st.session_state['ligand_smiles_valid'] = None
        st.session_state['ligand_protein_data'] = None
        st.session_state['prediction_complete'] = False  # Flag to track if prediction has been run

    # Check if we have existing prediction results in session state
    if st.session_state['prediction_complete'] and st.session_state['ligand_results'] and not predict_ligand_button:
        # Display results without running the prediction again
        results = st.session_state['ligand_results']
        protein_data = st.session_state['ligand_protein_data']

        st.subheader("🔝 Top 5 Predicted Ligands")
        st.dataframe(pd.DataFrame(results, columns=["SMILES", "Score"]))

        # Store ligand blocks and docking scores to avoid recalculating
        if 'ligand_blocks' not in st.session_state:
            st.session_state['ligand_blocks'] = {}
            st.session_state['docking_scores'] = {}

        for i, (smile, score) in enumerate(results, 1):
            with st.container():
                # Check if we've already processed this ligand
                if f"ligand_{i}" not in st.session_state['ligand_blocks']:
                    mol = Chem.MolFromSmiles(smile)
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol)
                    AllChem.MMFFOptimizeMolecule(mol)
                    ligand_block = Chem.MolToPDBBlock(mol)
                    st.session_state['ligand_blocks'][f"ligand_{i}"] = ligand_block

                    # Calculate docking score
                    docking_score = calculate_docking_score(protein_data, ligand_block)
                    st.session_state['docking_scores'][f"ligand_{i}"] = docking_score
                else:
                    ligand_block = st.session_state['ligand_blocks'][f"ligand_{i}"]
                    docking_score = st.session_state['docking_scores'][f"ligand_{i}"]

                st.markdown(f"### 🧪 Ligand #{i}")
                score_col1, score_col2 = st.columns(2)
                with score_col1:
                    st.info(f"**AI Score:** {score:.1f}%")
                with score_col2:
                    st.success(f"**Docking Score:** {docking_score:.2f} kcal/mol")

                viewer = py3Dmol.view(width=600, height=500)
                # Add protein model and color by user-selected scheme (model index 0)
                protein_style = {style_map.get(viz_style, 'cartoon'): {'colorscheme': color_map.get(color_scheme, 'chain'), 'opacity': 1}}
                viewer.addModel(protein_data, 'pdb')
                viewer.setStyle({'model': 0}, protein_style)
                # Add ligand model and color green (model index 1, fully opaque, stick style for visibility)
                viewer.addModel(ligand_block, 'pdb')
                viewer.setStyle({'model': 1}, {'stick': {'color': 'green', 'opacity': 1}})
                viewer.setBackgroundColor(bg_color_map.get(background_color, 'white'))
                viewer.zoomTo()

                # Apply spin setting from sidebar configuration
                if spin_model:
                    viewer.spin(True)

                st.markdown("#### Protein-Ligand Complex")
                st.markdown(f"Binding affinity: {docking_score:.2f} kcal/mol")
                components.html(viewer._make_html(), height=520)

                # Help text
                st.caption("👆 Tip: Click and drag to rotate. Scroll to zoom. Shift+click to translate.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        label=f"Download Ligand #{i} PDB",
                        data=ligand_block,
                        file_name=f"ligand_{i}.pdb",
                        mime="chemical/x-pdb",
                        key=f"download_ligand_{i}_pdb"
                    )
                with col2:
                    st.download_button(
                        label="Download Protein PDB",
                        data=protein_data,
                        file_name="protein.pdb",
                        mime="chemical/x-pdb",
                        key=f"download_protein_{i}_pdb"
                    )
                with col3:
                    complex_data = protein_data + "\n" + ligand_block
                    st.download_button(
                        label="Download Complex PDB",
                        data=complex_data,
                        file_name=f"complex_{i}.pdb",
                        mime="chemical/x-pdb",
                        key=f"download_complex_{i}_pdb"
                    )
        st.success(f"✅ Completed.")

    if predict_ligand_button and uploaded_protein:
        start = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            protein_pdb = tmp_path / "protein.pdb"
            protein_pdb.write_bytes(uploaded_protein.getvalue())
            sequence = extract_sequence_from_pdb(str(protein_pdb))

            if not sequence:
                st.error("Could not extract protein sequence from PDB.")
                st.stop()

            protein_tensor = torch.tensor(seq_cat(sequence), dtype=torch.long).to(device)

            # Ensure protein sequence is exactly 1000 amino acids as expected by the model
            if protein_tensor.size(0) < 1000:
                # Pad with zeros if shorter
                padded = torch.zeros(1000, dtype=torch.long, device=device)
                padded[:protein_tensor.size(0)] = protein_tensor
                protein_tensor = padded
            else:
                # Truncate if longer
                protein_tensor = protein_tensor[:1000]

            # Load ligand data based on selected data source
            if data_source == "Local CSV (data/kiba_test.csv)":
                try:
                    df = pd.read_csv(KIBA_DATA_PATH)
                    st.info(f"Using default KIBA test data with {len(df)} compounds")

                    # Show CSV columns to help the user understand the data structure
                    with st.expander("KIBA Test Data Structure"):
                        st.write("Columns in the KIBA test data:")
                        for col in df.columns:
                            st.write(f"- {col}")
                        st.write("The 'smiles' column will be used for ligand SMILES strings.")
                except Exception as e:
                    st.error(f"Error loading KIBA test data: {str(e)}")
                    st.stop()
            elif data_source == "PubChem":
                if not search_query:
                    st.error("Please enter a search term in the sidebar")
                    st.stop()
                df = search_pubchem(search_query, max_results)
                if df.empty:
                    st.error(f"No compounds found in PubChem for query: {search_query}")
                    st.stop()
                st.info(f"Found {len(df)} compounds in PubChem for query: {search_query}")
            elif data_source == "ChEMBL":
                if not search_query:
                    st.error("Please enter a search term in the sidebar")
                    st.stop()
                df = search_chembl(search_query, max_results)
                if df.empty:
                    st.error(f"No compounds found in ChEMBL for query: {search_query}")
                    st.stop()
                st.info(f"Found {len(df)} compounds in ChEMBL for query: {search_query}")
            elif data_source == "DrugBank (Approved Drugs)":
                df = get_drugbank_approved_drugs()
                st.info(f"Using {len(df)} approved drugs from DrugBank")

            # Find the SMILES column in the dataframe
            smiles_col = next((col for col in df.columns if 'smiles' in col.lower()), None)
            if not smiles_col:
                st.error("SMILES column not found in the data.")
                st.stop()

            # Show a preview of the compounds being used
            with st.expander("Preview compounds being analyzed"):
                st.dataframe(df.head(10))

            smiles_list = df[smiles_col].tolist()[:1000]
            data_list, smiles_valid = [], []
            skipped_ligands = []
            for smile in smiles_list:
                # Validate SMILES before processing
                mol = Chem.MolFromSmiles(smile)
                if mol is None:
                    skipped_ligands.append((smile, 'Invalid SMILES or not a SMILES string (possibly a name)'))
                    continue
                try:
                    c_size, features, edge_index = smile_to_graph(smile)
                    if c_size is None or features is None or edge_index is None:
                        skipped_ligands.append((smile, 'Invalid molecule graph'))
                        continue
                    x = torch.tensor(np.array(features), dtype=torch.float)
                    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                    batch = torch.zeros(x.size(0), dtype=torch.long)
                    data = Data(x=x, edge_index=edge_index, batch=batch)
                    data.target = protein_tensor.unsqueeze(0)
                    data_list.append(data)
                    smiles_valid.append(smile)
                except Exception as e:
                    skipped_ligands.append((smile, f"{str(e)} | features shape: {np.array(features).shape if 'features' in locals() else 'N/A'}"))
                    continue

            if skipped_ligands:
                with st.expander("⚠️ Skipped Ligands (click to view details)"):
                    for smile, reason in skipped_ligands:
                        st.write(f"SMILES: {smile} | Reason: {reason}")

            if not data_list:
                st.error("No valid ligand graphs could be processed. Please check your input data or try different ligands.")
                st.stop()

            # Process in smaller batches to avoid memory issues
            loader = DataLoader(data_list, batch_size=32)
            predictions = []

            with st.spinner("Running predictions..."):
                success_count = 0
                for idx, batch in enumerate(loader):
                    try:
                        batch = batch.to(device)
                        with torch.no_grad():
                            output = model(batch)
                            predictions.extend(output.cpu().numpy().flatten())
                        success_count += len(batch)
                        # Update progress
                        if idx % 5 == 0:
                            st.write(f"Processed {success_count}/{len(data_list)} ligands...")
                    except Exception as e:
                        st.warning(f"Skipped prediction for batch {idx} due to: {str(e)}")
                        continue

            results = sorted(zip(smiles_valid, predictions), key=lambda x: x[1], reverse=True)[:5]

            # Always use 'chain' color scheme for all protein visualizations
            color_scheme = 'Chain'

            st.session_state['ligand_results'] = results
            st.session_state['ligand_smiles_valid'] = smiles_valid
            st.session_state['ligand_protein_data'] = protein_data
            st.session_state['prediction_complete'] = True  # Set flag to true after prediction

            results = st.session_state.get('ligand_results')
            smiles_valid = st.session_state.get('ligand_smiles_valid')
            protein_data = st.session_state.get('ligand_protein_data')

            if results and smiles_valid and protein_data:
                st.subheader("🔝 Top 5 Predicted Ligands")
                st.dataframe(pd.DataFrame(results, columns=["SMILES", "Score"]))
                for i, (smile, score) in enumerate(results, 1):
                    with st.container():
                        mol = Chem.MolFromSmiles(smile)
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol)
                        AllChem.MMFFOptimizeMolecule(mol)
                        ligand_block = Chem.MolToPDBBlock(mol)
                        docking_score = calculate_docking_score(protein_data, ligand_block)
                        st.markdown(f"### 🧪 Ligand #{i}")
                        score_col1, score_col2 = st.columns(2)
                        with score_col1:
                            st.info(f"**AI Score:** {float(score):.4f}")
                        with score_col2:
                            st.success(f"**Docking Score:** {docking_score:.2f} kcal/mol")
                        viewer = py3Dmol.view(width=600, height=500)
                        # Add protein model and color by user-selected scheme (model index 0)
                        protein_style = {style_map.get(viz_style, 'cartoon'): {'colorscheme': color_map.get(color_scheme, 'chain'), 'opacity': 1}}
                        viewer.addModel(protein_data, 'pdb')
                        viewer.setStyle({'model': 0}, protein_style)
                        # Add ligand model and color green (model index 1, fully opaque, stick style for visibility)
                        viewer.addModel(ligand_block, 'pdb')
                        viewer.setStyle({'model': 1}, {'stick': {'color': 'green', 'opacity': 1}})
                        viewer.setBackgroundColor(bg_color_map.get(background_color, 'white'))
                        viewer.zoomTo()

                        # Apply spin setting from sidebar configuration
                        if spin_model:
                            viewer.spin(True)

                        st.markdown("#### Protein-Ligand Complex")
                        st.markdown(f"Binding affinity: {docking_score:.2f} kcal/mol")
                        components.html(viewer._make_html(), height=520)

                        # Help text
                        st.caption("👆 Tip: Click and drag to rotate. Scroll to zoom. Shift+click to translate.")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.download_button(
                                label=f"Download Ligand #{i} PDB",
                                data=ligand_block,
                                file_name=f"ligand_{i}.pdb",
                                mime="chemical/x-pdb",
                                key=f"download_ligand_{i}_pdb"
                            )
                        with col2:
                            st.download_button(
                                label="Download Protein PDB",
                                data=protein_data,
                                file_name="protein.pdb",
                                mime="chemical/x-pdb",
                                key=f"download_protein_{i}_pdb"
                            )
                        with col3:
                            complex_data = protein_data + "\n" + ligand_block
                            st.download_button(
                                label="Download Complex PDB",
                                data=complex_data,
                                file_name=f"complex_{i}.pdb",
                                mime="chemical/x-pdb",
                                key=f"download_complex_{i}_pdb"
                            )
        st.success(f"✅ Completed.")

        # Check if we have existing prediction results in session state
        if st.session_state['prediction_complete'] and st.session_state['ligand_results'] and not predict_ligand_button:
            # Display results without running the prediction again
            results = st.session_state['ligand_results']
            protein_data = st.session_state['ligand_protein_data']

            st.subheader("🔝 Top 5 Predicted Ligands")
            st.dataframe(pd.DataFrame(results, columns=["SMILES", "Score"]))

            # Store ligand blocks and docking scores to avoid recalculating
            if 'ligand_blocks' not in st.session_state:
                st.session_state['ligand_blocks'] = {}
                st.session_state['docking_scores'] = {}

            for i, (smile, score) in enumerate(results, 1):
                with st.container():
                    # Check if we've already processed this ligand
                    if f"ligand_{i}" not in st.session_state['ligand_blocks']:
                        mol = Chem.MolFromSmiles(smile)
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol)
                        AllChem.MMFFOptimizeMolecule(mol)
                        ligand_block = Chem.MolToPDBBlock(mol)
                        st.session_state['ligand_blocks'][f"ligand_{i}"] = ligand_block

                        # Calculate docking score
                        docking_score = calculate_docking_score(protein_data, ligand_block)
                        st.session_state['docking_scores'][f"ligand_{i}"] = docking_score
                    else:
                        ligand_block = st.session_state['ligand_blocks'][f"ligand_{i}"]
                        docking_score = st.session_state['docking_scores'][f"ligand_{i}"]

                    st.markdown(f"### 🧪 Ligand #{i}")
                    score_col1, score_col2 = st.columns(2)
                    with score_col1:
                        # Scale the score to be between 0 and 17 (KIBA dataset range)
                        scaled_score = min(max(float(score), 0), 17)  # Ensure score is within 0-17 range
                        # Convert to percentage (0-17 → 0-100%)
                        percentage_score = (scaled_score / 17) * 100
                        st.info(f"**AI Score:** {percentage_score:.1f}%")
                    with score_col2:
                        st.success(f"**Docking Score:** {docking_score:.2f} kcal/mol")

                    viewer = py3Dmol.view(width=600, height=500)
                    # Add protein model and color by user-selected scheme (model index 0)
                    protein_style = {style_map.get(viz_style, 'cartoon'): {'colorscheme': color_map.get(color_scheme, 'chain'), 'opacity': 1}}
                    viewer.addModel(protein_data, 'pdb')
                    viewer.setStyle({'model': 0}, protein_style)
                    # Add ligand model and color green (model index 1, fully opaque, stick style for visibility)
                    viewer.addModel(ligand_block, 'pdb')
                    viewer.setStyle({'model': 1}, {'stick': {'color': 'green', 'opacity': 1}})
                    viewer.setBackgroundColor(bg_color_map.get(background_color, 'white'))
                    viewer.zoomTo()

                    # Apply spin setting from sidebar configuration
                    if spin_model:
                        viewer.spin(True)

                    st.markdown("#### Protein-Ligand Complex")
                    st.markdown(f"Binding affinity: {docking_score:.2f} kcal/mol")
                    components.html(viewer._make_html(), height=520)

                    # Help text
                    st.caption("👆 Tip: Click and drag to rotate. Scroll to zoom. Shift+click to translate.")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            label=f"Download Ligand #{i} PDB",
                            data=ligand_block,
                            file_name=f"ligand_{i}.pdb",
                            mime="chemical/x-pdb",
                            key=f"download_ligand_{i}_pdb"
                        )
                    with col2:
                        st.download_button(
                            label="Download Protein PDB",
                            data=protein_data,
                            file_name="protein.pdb",
                            mime="chemical/x-pdb",
                            key=f"download_protein_{i}_pdb"
                        )
                    with col3:
                        complex_data = protein_data + "\n" + ligand_block
                        st.download_button(
                            label="Download Complex PDB",
                            data=complex_data,
                            file_name=f"complex_{i}.pdb",
                            mime="chemical/x-pdb",
                            key=f"download_complex_{i}_pdb"
                        )
                st.success(f"✅ Completed.")

with tab2:
    st.title("🧬 Predict Protein-Protein Interactions")

    st.markdown("""
    ### Find proteins that can interact with your uploaded protein

    Upload a protein structure (PDB file) and we'll predict the top 5 proteins
    that are most likely to interact with it. For each prediction, we'll show:

    - The predicted protein partner
    - Interaction confidence score
    - 3D visualization of the potential complex
    - Downloadable PDB files of the complex
    """)

    uploaded_target_protein = st.file_uploader("🔬 Upload Target Protein PDB File", type=["pdb"], key="protein_tab")

    # Display preview of uploaded protein
    if uploaded_target_protein is not None:
        st.subheader("🔍 Target Protein Structure")

        # Process uploaded protein file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
            tmp.write(uploaded_target_protein.getvalue())
            target_pdb_path = tmp.name

        with open(target_pdb_path, 'r') as f:
            target_protein_data = f.read()

        # Create 3D viewer for target protein
        viewer = py3Dmol.view(width=600, height=400)
        viewer.addModel(target_protein_data, 'pdb')
        viewer.setStyle({}, {'cartoon': {'color': 'spectrum'}})
        viewer.setBackgroundColor('white')
        viewer.zoomTo()
        viewer.spin(True)
        components.html(viewer._make_html(), height=400)

        predict_protein_button = st.button("🔍 Predict Top 5 Interacting Proteins")

        if predict_protein_button:
            # Initialize the protein interaction predictor
            predictor = ProteinInteractionPredictor()

            with st.spinner("Predicting protein interactions..."):
                # Predict top 5 protein interactions
                predictions = predictor.predict_interactions(target_pdb_path, num_predictions=5)

                # Create a dataframe of predictions for display
                pred_df = pd.DataFrame(predictions)

                # Display the prediction results table
                st.subheader("🔝 Top 5 Predicted Protein Interactions")
                st.dataframe(
                    pred_df[["protein", "pdb_id", "description", "confidence"]],
                    column_config={
                        "protein": "Protein Name",
                        "pdb_id": "PDB ID",
                        "description": "Description",
                        "confidence": "Confidence"
                    }
                )

                # Visualize each predicted complex
                for i, pred in enumerate(predictions, 1):
                    st.markdown(f"### 🧬 Complex #{i}: {pred['protein']} — Confidence: {pred['confidence']}")

                    # Download the partner protein PDB
                    with st.spinner(f"Downloading {pred['protein']} (PDB ID: {pred['pdb_id']})..."):
                        partner_pdb_data = download_pdb(pred['pdb_id'])

                    if partner_pdb_data:
                        # Create a protein complex model
                        with st.spinner("Creating protein complex..."):
                            complex_pdb_data = create_protein_complex(target_protein_data, partner_pdb_data)

                        # Create 3D visualization of the complex with enhanced styling
                        viewer = py3Dmol.view(width=600, height=500)
                        viewer.addModel(complex_pdb_data, 'pdb')

                        # Color the chains differently to distinguish the two proteins
                        chains = {}
                        for line in complex_pdb_data.splitlines():
                            if line.startswith("ATOM") or line.startswith("HETATM"):
                                chain_id = line[21]
                                chains[chain_id] = True

                        chain_ids = list(chains.keys())
                        if len(chain_ids) > 1:
                            # First half of chains = target protein
                            target_chains = chain_ids[:len(chain_ids)//2]
                            # Second half of chains = partner protein
                            partner_chains = chain_ids[len(chain_ids)//2:]

                            # Style target protein - cartoon with transparency
                            viewer.setStyle({'chain': ','.join(target_chains)},
                                           {'cartoon': {'color': 'lightblue', 'opacity': 0.8}})

                            # Add surface to target protein with high transparency
                            viewer.addSurface(py3Dmol.VDW,
                                             {'opacity': 0.3, 'color': 'lightblue'},
                                             {'chain': ','.join(target_chains)})

                            # Style partner protein - cartoon with different color
                            viewer.setStyle({'chain': ','.join(partner_chains)},
                                           {'cartoon': {'color': 'pink', 'opacity': 0.8}})

                            # Add surface to partner protein with high transparency
                            viewer.addSurface(py3Dmol.VDW,
                                             {'opacity': 0.3, 'color': 'pink'},
                                             {'chain': ','.join(partner_chains)})

                            # Highlight the interface region
                            viewer.setStyle({'chain': ','.join(target_chains)},
                                           {'cartoon': {'color': 'lightblue'}})
                            viewer.setStyle({'chain': ','.join(partner_chains)},
                                           {'cartoon': {'color': 'salmon'}})

                            # Add labels for the proteins
                            target_label = "Your Protein"
                            partner_label = pred['protein']

                            # Calculate positions for labels (near centers of proteins)
                            viewer.addLabel(target_label,
                                           {'fontColor': 'black', 'backgroundColor': 'lightblue', 'fontOpacity': 0.7},
                                           {'chain': str(target_chains[0])})
                            viewer.addLabel(partner_label,
                                           {'fontColor': 'black', 'backgroundColor': 'salmon', 'fontOpacity': 0.7},
                                           {'chain': str(partner_chains[0])})
                        else:
                            # Fallback if chain identification fails
                            viewer.setStyle({}, {'cartoon': {'colorscheme': 'spectrum'}})

                        # Set viewer properties
                        viewer.setBackgroundColor('white')
                        viewer.zoomTo()

                        # Add spin and animation controls
                        spin_option = st.checkbox(f"Spin Complex #{i}", value=True)
                        if spin_option:
                            viewer.spin(True)

                        # Calculate and display docking score
                        docking_score = calculate_docking_score(target_protein_data, partner_pdb_data)
                        st.success(f"Docking Score: {docking_score:.3f}")

                        # Display Ligand AI Score if available
                        if 'affinity' in pred:
                            st.info(f"Ligand AI Score (Predicted Affinity): {pred['affinity']:.3f}")

                        # Cartoon view for the complex (all chains)
                        viewer.setStyle({}, {"cartoon": {"color": "spectrum", "opacity": 0.8}})
                        viewer.zoomTo()
                        viewer.spin(True)

                        # Add slicing option
                        slice_option = st.checkbox(f"Show Sliced View for Complex #{i}", value=False)
                        if slice_option:
                            viewer.addSlab({'x': 0, 'y': 0, 'z': 0}, {'normal': {'x': 1, 'y': 0, 'z': 0}})

                        # Show the complex visualization
                        components.html(viewer._make_html(), height=550)

                        # Download buttons for the complex
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label=f"Download Complex PDB",
                                data=complex_pdb_data,
                                file_name=f"complex_{pred['pdb_id']}.pdb",
                                mime="chemical/x-pdb",
                                key=f"download_complex_{pred['pdb_id']}_pdb"
                            )
                        with col2:
                            st.download_button(
                                label=f"Download Partner ({pred['pdb_id']}) PDB",
                                data=partner_pdb_data,
                                file_name=f"{pred['pdb_id']}.pdb",
                                mime="chemical/x-pdb",
                                key=f"download_partner_{pred['pdb_id']}_pdb"
                            )
                    else:
                        st.error(f"Could not download PDB file for {pred['protein']} (PDB ID: {pred['pdb_id']})")

    # Clean up the temporary file
    try:
        os.unlink(target_pdb_path)
    except:
        pass
