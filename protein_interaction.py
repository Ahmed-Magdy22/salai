import streamlit as st
import torch
import numpy as np
import os
import tempfile
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Superimposer
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
import py3Dmol
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem

# Define a set of common proteins with PDB IDs that can be used for interaction prediction
COMMON_PROTEINS = [
    {"name": "Ubiquitin", "pdb_id": "1UBQ", "description": "Involved in protein degradation"},
    {"name": "Lysozyme", "pdb_id": "1LYZ", "description": "Antibacterial enzyme"},
    {"name": "Hemoglobin", "pdb_id": "1HHO", "description": "Oxygen-carrying protein"},
    {"name": "Insulin", "pdb_id": "4INS", "description": "Hormone regulating glucose metabolism"},
    {"name": "Cytochrome C", "pdb_id": "1HRC", "description": "Electron transport protein"},
    {"name": "Green Fluorescent Protein", "pdb_id": "1EMA", "description": "Fluorescent protein from jellyfish"},
    {"name": "CRISPR Cas9", "pdb_id": "4OO8", "description": "RNA-guided DNA endonuclease"},
    {"name": "p53", "pdb_id": "2OCJ", "description": "Tumor suppressor protein"},
    {"name": "Beta-Lactamase", "pdb_id": "1BTL", "description": "Enzyme that provides antibiotic resistance"},
    {"name": "Collagen", "pdb_id": "1CAG", "description": "Structural protein in connective tissues"},
    {"name": "Actin", "pdb_id": "1ATN", "description": "Cytoskeletal protein"},
    {"name": "Albumin", "pdb_id": "1AO6", "description": "Transport protein in blood plasma"},
    {"name": "Integrin", "pdb_id": "1JV2", "description": "Cell adhesion receptor"},
    {"name": "Keratin", "pdb_id": "3TNU", "description": "Structural protein in hair, nails, and skin"},
    {"name": "Elastin", "pdb_id": "2JF9", "description": "Elastic protein in connective tissue"},
    {"name": "Myosin", "pdb_id": "1DFK", "description": "Motor protein involved in muscle contraction"},
    {"name": "Immunoglobulin G", "pdb_id": "1IGT", "description": "Antibody"},
    {"name": "Tau Protein", "pdb_id": "6QJH", "description": "Microtubule-associated protein in neurons"},
    {"name": "Calmodulin", "pdb_id": "1CLL", "description": "Calcium-binding messenger protein"},
    {"name": "Protease", "pdb_id": "5I7X", "description": "Enzyme that breaks down proteins"}
]

# Class to handle protein-protein interaction prediction
class ProteinInteractionPredictor:
    def __init__(self):
        # This is a simplified model that uses structural similarity and complementarity
        # for a real application, you'd use a proper PPI prediction model
        pass

    def predict_interactions(self, target_pdb_path, num_predictions=5):
        """
        Predict proteins that can interact with the target protein

        Args:
            target_pdb_path: Path to the target protein PDB file
            num_predictions: Number of interaction predictions to return

        Returns:
            List of dictionaries with interaction predictions and scores
        """
        # In a real application, this would use a machine learning model trained on protein interactions
        # For this demo, we'll use a simple scoring function based on protein properties

        # Parse the target protein
        parser = PDBParser(QUIET=True)
        target_structure = parser.get_structure("target", target_pdb_path)

        # Extract basic features from the target protein
        target_size = self._count_residues(target_structure)
        target_chains = len(list(target_structure.get_chains()))

        # Simulate prediction scores for the common proteins
        results = []
        for protein in COMMON_PROTEINS:
            # Calculate a similarity score (this is a simplified mock scoring function)
            # In a real application, you would use ML models or docking algorithms
            interaction_score = self._calculate_mock_score(protein, target_size, target_chains)

            results.append({
                "protein": protein["name"],
                "pdb_id": protein["pdb_id"],
                "description": protein["description"],
                "interaction_score": interaction_score,
                "confidence": f"{min(interaction_score * 100, 99.9):.1f}%"
            })

        # Sort by interaction score (higher is better)
        results.sort(key=lambda x: x["interaction_score"], reverse=True)

        # Return top predictions
        return results[:num_predictions]

    def _count_residues(self, structure):
        """Count the number of residues in a protein structure"""
        count = 0
        for model in structure:
            for chain in model:
                count += len(list(chain.get_residues()))
        return count

    def _calculate_mock_score(self, protein, target_size, target_chains):
        """
        Calculate a mock interaction score for demonstration purposes
        In a real application, you would use actual features and a trained model
        """
        # Generate a plausible but random score between 0.5 and 0.95
        # This is just for demonstration - real scores would come from a proper model
        base_score = 0.5 + (hash(protein["name"] + protein["pdb_id"]) % 1000) / 2000

        # Add small random variation
        variation = random.uniform(-0.05, 0.05)

        return min(max(base_score + variation, 0.5), 0.95)

def download_pdb(pdb_id):
    """
    Download a PDB file from the RCSB PDB database
    """
    import requests
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        st.error(f"Failed to download PDB file for {pdb_id}. Status code: {response.status_code}")
        return None

def create_protein_complex(target_pdb_data, partner_pdb_data):
    """
    Create a mock protein complex by combining two protein structures
    In a real application, you would use protein docking software

    Args:
        target_pdb_data: String containing the target protein's PDB data
        partner_pdb_data: String containing the partner protein's PDB data

    Returns:
        String containing the combined PDB data with the partner positioned next to the target
    """
    # In a real application, you would use docking software to predict the binding orientation
    # For this demo, we'll just position the second protein next to the first one

    # Write the PDB data to temporary files
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as target_file:
        target_file.write(target_pdb_data.encode())
        target_path = target_file.name

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as partner_file:
        partner_file.write(partner_pdb_data.encode())
        partner_path = partner_file.name

    # Parse the structures
    parser = PDBParser(QUIET=True)
    target_structure = parser.get_structure("target", target_path)
    partner_structure = parser.get_structure("partner", partner_path)

    # Get the center of mass for both structures
    target_com = calculate_center_of_mass(target_structure)
    partner_com = calculate_center_of_mass(partner_structure)

    # Calculate translation vector to position partner next to target
    # with a small gap between them
    direction = np.array([1.0, 0.0, 0.0])  # arbitrary direction
    offset = 10.0  # Angstroms - distance between the structures
    translation = target_com + direction * offset - partner_com

    # Translate the partner structure
    for atom in partner_structure.get_atoms():
        atom.transform(np.identity(3), translation)

    # Create a new structure with both proteins
    complex_structure = Structure("complex")
    complex_model = Model(0)
    complex_structure.add(complex_model)

    # Get existing chain IDs from target to avoid conflicts
    existing_chain_ids = set()
    for chain in target_structure.get_chains():
        existing_chain_ids.add(chain.id)

    # Add target chains first
    for chain in target_structure.get_chains():
        new_chain = Chain(chain.id)
        for residue in chain:
            new_chain.add(residue)
        complex_model.add(new_chain)

    # Add partner chains with guaranteed unique IDs
    available_ids = [chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) not in existing_chain_ids]
    if len(available_ids) < len(list(partner_structure.get_chains())):
        # If we need more IDs, add numeric ones
        available_ids.extend([str(i) for i in range(1, 10)])

    for i, chain in enumerate(partner_structure.get_chains()):
        if i < len(available_ids):
            new_id = available_ids[i]
        else:
            # Fallback to numeric chain IDs if we run out of letters
            new_id = str((i - len(available_ids)) % 9 + 1)

        new_chain = Chain(new_id)
        for residue in chain:
            new_chain.add(residue)
        complex_model.add(new_chain)

    # Write the complex to a PDB file
    io = PDBIO()
    io.set_structure(complex_structure)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as complex_file:
        io.save(complex_file.name)
        with open(complex_file.name, 'r') as f:
            complex_pdb_data = f.read()

    # Clean up
    os.unlink(target_path)
    os.unlink(partner_path)
    os.unlink(complex_file.name)

    return complex_pdb_data

def calculate_center_of_mass(structure):
    """Calculate the center of mass of a protein structure"""
    coords = []
    masses = []

    for atom in structure.get_atoms():
        coords.append(atom.coord)
        # Use a uniform mass for simplicity
        masses.append(1.0)

    coords = np.array(coords)
    masses = np.array(masses)

    total_mass = np.sum(masses)
    center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass

    return center_of_mass

def calculate_docking_score(target_pdb_data, ligand_pdb_data):
    """
    Calculate a mock docking score between a protein and ligand

    In a real application, you would use a proper docking program like AutoDock Vina
    This is a simplified calculation for demonstration purposes

    Args:
        target_pdb_data: String containing the target protein's PDB data
        ligand_pdb_data: String containing the ligand's PDB data

    Returns:
        A docking score (lower is better, like real docking software)
    """
    # This is a simplified scoring function for demonstration
    # In a real application, you'd use AutoDock Vina or similar software

    # Count the atoms in the ligand as a rough size estimate
    ligand_atom_count = sum(1 for line in ligand_pdb_data.splitlines()
                          if line.startswith("ATOM") or line.startswith("HETATM"))

    # Calculate a mock binding energy based on ligand size
    # Larger ligands often have more binding possibilities but may have steric issues
    if ligand_atom_count < 20:
        base_score = -5.5  # Small molecules often have weak binding
    elif ligand_atom_count < 40:
        base_score = -7.2  # Medium molecules often have moderate binding
    else:
        base_score = -8.5  # Large molecules often have stronger binding

    # Add some randomness to simulate the variety of real docking scores
    import random
    variation = random.uniform(-1.5, 1.5)

    # Calculate final score (negative numbers, lower is better)
    docking_score = base_score + variation

    return docking_score
