#!/usr/bin/env python3
"""
Script to check RDKit installation and provide installation guidance
"""

import sys
import platform

def check_rdkit():
    print("=== RDKit Installation Check ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print()
    
    # Check basic RDKit import
    try:
        from rdkit import Chem
        print("✅ RDKit Chem module: Available")
        
        # Test basic functionality
        mol = Chem.MolFromSmiles("CCO")
        if mol:
            print("✅ SMILES parsing: Working")
        else:
            print("❌ SMILES parsing: Failed")
            
    except ImportError as e:
        print(f"❌ RDKit Chem module: Not available - {e}")
        return False
    
    # Check AllChem (needed for 3D coordinates)
    try:
        from rdkit.Chem import AllChem
        print("✅ RDKit AllChem module: Available")
        
        # Test 3D embedding
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol)
        if result == 0:
            print("✅ 3D molecule embedding: Working")
        else:
            print("⚠️  3D molecule embedding: May have issues")
            
    except ImportError as e:
        print(f"❌ RDKit AllChem module: Not available - {e}")
        print("\n🔧 Fix: Try installing RDKit with conda:")
        print("   conda install -c conda-forge rdkit")
        return False
    
    # Check drawing functionality (optional)
    try:
        from rdkit.Chem import Draw
        print("✅ RDKit Draw module: Available")
    except ImportError:
        print("⚠️  RDKit Draw module: Not available (optional)")
        if platform.system() == "Linux":
            print("   This might be due to missing graphics libraries")
            print("   Try: sudo apt-get install libxrender1 libxext6")
    
    # Check PandasTools (optional)
    try:
        from rdkit.Chem import PandasTools
        print("✅ RDKit PandasTools: Available")
    except ImportError:
        print("⚠️  RDKit PandasTools: Not available (optional)")
    
    return True

def provide_installation_guide():
    print("\n=== Installation Guide ===")
    
    if platform.system() == "Darwin":  # macOS
        print("For macOS:")
        print("1. Install via conda (recommended):")
        print("   conda install -c conda-forge rdkit")
        print("\n2. Or via pip:")
        print("   pip install rdkit")
        
    elif platform.system() == "Linux":
        print("For Linux:")
        print("1. Install system dependencies:")
        print("   sudo apt-get update")
        print("   sudo apt-get install libxrender1 libxext6 libsm6 libxrandr2 libfontconfig1 libxss1")
        print("\n2. Install RDKit via conda (recommended):")
        print("   conda install -c conda-forge rdkit")
        print("\n3. Or via pip:")
        print("   pip install rdkit")
        
    elif platform.system() == "Windows":
        print("For Windows:")
        print("1. Install via conda (recommended):")
        print("   conda install -c conda-forge rdkit")
        print("\n2. Or via pip:")
        print("   pip install rdkit")
    
    print("\n=== For this application ===")
    print("The core functionality only requires:")
    print("- rdkit.Chem (for SMILES parsing)")
    print("- rdkit.Chem.AllChem (for 3D coordinates)")
    print("\nDrawing functionality is optional and not critical.")

if __name__ == "__main__":
    rdkit_ok = check_rdkit()
    
    if not rdkit_ok:
        provide_installation_guide()
    else:
        print("\n✅ RDKit is properly installed and working!")
        print("The protein-ligand interaction app should work correctly.")
