import streamlit as st
import pandas as pd
import requests
import time
import urllib.parse
from rdkit import Chem
# Import PandasTools conditionally to avoid libXrender.so.1 error
try:
    from rdkit.Chem import PandasTools
    HAS_RDKIT_DRAW = True
except ImportError:
    HAS_RDKIT_DRAW = False
    st.warning("RDKit drawing functionality is not available. Some visualization features may be limited.")
import io

def search_pubchem(query, max_compounds=100):
    """
    Search PubChem for compounds matching the query.

    Args:
        query: Search term (name, SMILES, etc.)
        max_compounds: Maximum number of compounds to return

    Returns:
        DataFrame with compound information
    """
    try:
        st.info(f"Searching PubChem for: {query}")

        # Common drug name to SMILES mapping for faster resolution of common searches
        common_drugs = {
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "acetaminophen": "CC(=O)NC1=CC=C(C=C1)O",
            "paracetamol": "CC(=O)NC1=CC=C(C=C1)O",  # same as acetaminophen
            "naproxen": "CC(C)C1=CC2=C(C=C1)C=C(C=C2)C(C)C(=O)O"
        }

        # If query is a common drug name, use the predefined SMILES
        if query.lower() in common_drugs:
            st.info(f"Using predefined SMILES for {query}")
            compounds = [{
                "cid": 0,  # Placeholder CID
                "compound_iso_smiles": common_drugs[query.lower()],
                "name": query.capitalize(),
                "formula": "",  # Could add formulas if needed
                "source": "PubChem (Common Drug)"
            }]
            return pd.DataFrame(compounds)

        # First, assume it's a name/keyword (don't try SMILES parsing first)
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_query}/cids/JSON"
        response = requests.get(search_url)

        # If name search works, process the results
        if response.status_code == 200:
            try:
                data = response.json()
                if "IdentifierList" in data and "CID" in data["IdentifierList"]:
                    cids = data["IdentifierList"]["CID"][:max_compounds]
                    compounds = []

                    for cid in cids:
                        # Get compound properties
                        prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,IUPACName,MolecularFormula/JSON"
                        prop_response = requests.get(prop_url)

                        if prop_response.status_code == 200:
                            prop_data = prop_response.json()
                            prop = prop_data["PropertyTable"]["Properties"][0]

                            compounds.append({
                                "cid": cid,
                                "compound_iso_smiles": prop.get("CanonicalSMILES", ""),
                                "name": prop.get("IUPACName", f"Compound {cid}"),
                                "formula": prop.get("MolecularFormula", ""),
                                "source": "PubChem"
                            })

                            # Be nice to the PubChem servers
                            time.sleep(0.1)

                    if compounds:
                        return pd.DataFrame(compounds)
            except Exception as e:
                st.warning(f"Error processing name search results: {str(e)}")

        # If name search fails, only then try as SMILES
        is_valid_smiles = False
        try:
            mol = Chem.MolFromSmiles(query)
            is_valid_smiles = mol is not None
        except Exception as e:
            is_valid_smiles = False
            st.warning(f"Not a valid SMILES string: {str(e)}")

        if is_valid_smiles:
            # URL encode the SMILES string for API request
            encoded_smiles = urllib.parse.quote(query)
            search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/cids/JSON"
            response = requests.get(search_url)

            if response.status_code == 200:
                data = response.json()
                if "IdentifierList" in data and "CID" in data["IdentifierList"]:
                    cids = data["IdentifierList"]["CID"][:max_compounds]
                    compounds = []

                    for cid in cids:
                        # Get compound properties
                        prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,IUPACName,MolecularFormula/JSON"
                        prop_response = requests.get(prop_url)

                        if prop_response.status_code == 200:
                            prop_data = prop_response.json()
                            prop = prop_data["PropertyTable"]["Properties"][0]

                            compounds.append({
                                "cid": cid,
                                "compound_iso_smiles": prop.get("CanonicalSMILES", ""),
                                "name": prop.get("IUPACName", f"Compound {cid}"),
                                "formula": prop.get("MolecularFormula", ""),
                                "source": "PubChem"
                            })

                            # Be nice to the PubChem servers
                            time.sleep(0.1)

                    if compounds:
                        return pd.DataFrame(compounds)

        # If all else fails, use default compounds
        st.warning(f"No compounds found for query: {query}. Using default compounds.")
        return get_default_compounds("aspirin")

    except Exception as e:
        st.error(f"Error searching PubChem: {str(e)}")
        return get_default_compounds("aspirin")

def search_chembl(query, max_compounds=100):
    """
    Search ChEMBL for compounds matching the query.

    Args:
        query: Search term (name, SMILES, etc.)
        max_compounds: Maximum number of compounds to return

    Returns:
        DataFrame with compound information
    """
    try:
        st.info(f"Searching ChEMBL for: {query}")

        # Common drug name to SMILES mapping for faster resolution of common searches
        common_drugs = {
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "acetaminophen": "CC(=O)NC1=CC=C(C=C1)O",
            "paracetamol": "CC(=O)NC1=CC=C(C=C1)O",  # same as acetaminophen
            "naproxen": "CC(C)C1=CC2=C(C=C1)C=C(C=C2)C(C)C(=O)O"
        }

        # If query is a common drug name, use the predefined SMILES
        if query.lower() in common_drugs:
            st.info(f"Using predefined SMILES for {query}")
            compounds = [{
                "chembl_id": "COMMON_DRUG",
                "compound_iso_smiles": common_drugs[query.lower()],
                "name": query.capitalize(),
                "formula": "",  # Could add formulas if needed
                "source": "ChEMBL (Common Drug)"
            }]
            return pd.DataFrame(compounds)

        # ChEMBL API URL
        base_url = "https://www.ebi.ac.uk/chembl/api/data"

        # First try as name/keyword search without SMILES parsing
        encoded_query = urllib.parse.quote(query)
        search_url = f"{base_url}/molecule?pref_name__contains={encoded_query}&format=json"
        response = requests.get(search_url)

        if response.status_code == 200:
            data = response.json()
            molecules = data.get("molecules", [])

            if molecules:
                compounds = []
                for mol in molecules[:max_compounds]:
                    try:
                        struct = mol.get("molecule_structures", {})
                        smiles = struct.get("canonical_smiles", "")

                        if smiles:
                            compounds.append({
                                "chembl_id": mol.get("molecule_chembl_id", ""),
                                "compound_iso_smiles": smiles,
                                "name": mol.get("pref_name", f"ChEMBL Compound"),
                                "formula": mol.get("molecule_properties", {}).get("full_molformula", ""),
                                "source": "ChEMBL"
                            })
                    except Exception as e:
                        continue

                if compounds:
                    return pd.DataFrame(compounds)

        # Try as SMILES only after name search fails
        is_valid_smiles = False
        try:
            mol = Chem.MolFromSmiles(query)
            is_valid_smiles = mol is not None
        except Exception as e:
            is_valid_smiles = False
            st.warning(f"Not a valid SMILES string: {str(e)}")

        if is_valid_smiles:
            encoded_smiles = urllib.parse.quote(query)
            search_url = f"{base_url}/molecule?molecule_structures__canonical_smiles__flexmatch={encoded_smiles}&format=json"
            response = requests.get(search_url)

            if response.status_code == 200:
                data = response.json()
                molecules = data.get("molecules", [])

                if molecules:
                    compounds = []
                    for mol in molecules[:max_compounds]:
                        try:
                            struct = mol.get("molecule_structures", {})
                            smiles = struct.get("canonical_smiles", "")

                            if smiles:
                                compounds.append({
                                    "chembl_id": mol.get("molecule_chembl_id", ""),
                                    "compound_iso_smiles": smiles,
                                    "name": mol.get("pref_name", f"ChEMBL Compound"),
                                    "formula": mol.get("molecule_properties", {}).get("full_molformula", ""),
                                    "source": "ChEMBL"
                                })
                        except Exception as e:
                            continue

                    if compounds:
                        return pd.DataFrame(compounds)

        # If all searches fail, try a more general search
        search_url = f"{base_url}/molecule?molecule_properties__full_molformula__contains=C&limit=10&format=json"
        response = requests.get(search_url)

        if response.status_code == 200:
            data = response.json()
            molecules = data.get("molecules", [])

            if molecules:
                compounds = []
                for mol in molecules[:max_compounds]:
                    try:
                        struct = mol.get("molecule_structures", {})
                        smiles = struct.get("canonical_smiles", "")

                        if smiles:
                            compounds.append({
                                "chembl_id": mol.get("molecule_chembl_id", ""),
                                "compound_iso_smiles": smiles,
                                "name": mol.get("pref_name", f"ChEMBL Compound"),
                                "formula": mol.get("molecule_properties", {}).get("full_molformula", ""),
                                "source": "ChEMBL"
                            })
                    except Exception as e:
                        continue

                if compounds:
                    return pd.DataFrame(compounds)

        # Still no results, return defaults
        st.warning(f"No compounds found in ChEMBL for: {query}. Using default compounds.")
        return get_default_compounds()

    except Exception as e:
        st.error(f"Error searching ChEMBL: {str(e)}")
        return get_default_compounds()

def combine_datasets(dfs):
    """Combine multiple DataFrames into one, ensuring the SMILES column is consistent"""
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Make sure we have a compound_iso_smiles column
    if "compound_iso_smiles" not in combined.columns and "SMILES" in combined.columns:
        combined = combined.rename(columns={"SMILES": "compound_iso_smiles"})
    elif "compound_iso_smiles" not in combined.columns and "smiles" in combined.columns:
        combined = combined.rename(columns={"smiles": "compound_iso_smiles"})

    # Drop any rows with empty SMILES
    combined = combined[combined["compound_iso_smiles"].notna() & (combined["compound_iso_smiles"] != "")]

    # Drop duplicates based on SMILES
    combined = combined.drop_duplicates(subset=["compound_iso_smiles"])

    return combined

def get_drugbank_approved_drugs(max_compounds=200):
    """Return a small dataset of approved drugs from DrugBank"""
    # This is a limited set of approved drugs with their SMILES
    # In a production app, you'd want to integrate with the full DrugBank API
    approved_drugs = [
        {"name": "Aspirin", "compound_iso_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "source": "DrugBank"},
        {"name": "Ibuprofen", "compound_iso_smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "source": "DrugBank"},
        {"name": "Acetaminophen", "compound_iso_smiles": "CC(=O)NC1=CC=C(C=C1)O", "source": "DrugBank"},
        {"name": "Atorvastatin", "compound_iso_smiles": "CC(C)C1=C(C=CC=C1)C(=O)NC(CC(C)C)C(=O)NC(CC2=CC=C(C=C2)F)C(O)CC(O)CC3=CC=C(C=C3)O", "source": "DrugBank"},
        {"name": "Simvastatin", "compound_iso_smiles": "CCC(C)(C)C1=C(C=C(C=C1)C2=CC=C(C=C2)C(C)C(=O)NC3CC4=C(C3)C=CC=C4)C5=CC=C(C=C5)O", "source": "DrugBank"},
        {"name": "Lisinopril", "compound_iso_smiles": "NCCCCC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C(C(C)CC)NC)C(=O)O", "source": "DrugBank"},
        {"name": "Metformin", "compound_iso_smiles": "CN(C)C(=N)NC(=N)N", "source": "DrugBank"},
        {"name": "Warfarin", "compound_iso_smiles": "CC(=O)CC(C1=CC=CC=C1)C2=C(C=C(C=C2)O)C(=O)O", "source": "DrugBank"},
        {"name": "Omeprazole", "compound_iso_smiles": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC", "source": "DrugBank"},
        {"name": "Fluoxetine", "compound_iso_smiles": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F", "source": "DrugBank"},
    ]
    return pd.DataFrame(approved_drugs)

def get_default_compounds(query_type="general"):
    """Return a set of default compounds when API searches fail"""
    if query_type == "aspirin" or query_type == "analgesic":
        # Return common analgesics/NSAIDs
        compounds = [
            {"name": "Aspirin", "compound_iso_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "source": "Default"},
            {"name": "Ibuprofen", "compound_iso_smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "source": "Default"},
            {"name": "Acetaminophen", "compound_iso_smiles": "CC(=O)NC1=CC=C(C=C1)O", "source": "Default"},
            {"name": "Naproxen", "compound_iso_smiles": "CC(C)C1=CC2=C(C=C1)C=C(C=C2)C(C)C(=O)O", "source": "Default"},
            {"name": "Diclofenac", "compound_iso_smiles": "CN1C=C(C=CC1=O)CC2=CC=CC=C2NC3=C(C=CC=C3Cl)Cl", "source": "Default"},
        ]
    else:
        # General drug compounds
        compounds = [
            {"name": "Aspirin", "compound_iso_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "source": "Default"},
            {"name": "Ibuprofen", "compound_iso_smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "source": "Default"},
            {"name": "Acetaminophen", "compound_iso_smiles": "CC(=O)NC1=CC=C(C=C1)O", "source": "Default"},
            {"name": "Metformin", "compound_iso_smiles": "CN(C)C(=N)NC(=N)N", "source": "Default"},
            {"name": "Atorvastatin", "compound_iso_smiles": "CC(C)C1=C(C=CC=C1)C(=O)NC(CC(C)C)C(=O)NC(CC2=CC=C(C=C2)F)C(O)CC(O)CC3=CC=C(C=C3)O", "source": "Default"},
        ]

    return pd.DataFrame(compounds)
