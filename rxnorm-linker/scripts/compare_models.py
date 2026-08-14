import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Paths to input files
INPUT_DIR = ROOT.parent.parent / "input"
SUB_0704 = ROOT.parent.parent / "submission" / "0704" / "03"  # MiniLM
SUB_0705 = ROOT.parent.parent / "submission" / "0705" / "01"  # SapBERT

# Load cached embeddings
EMBED_OLD = ROOT / "data" / "rxnorm" / "cache" / "rxnorm_embed_old.npz"  # MiniLM 384d
EMBED_NEW = ROOT / "data" / "rxnorm" / "cache" / "rxnorm_embed.npz"      # SapBERT 768d

print("Loading embeddings...")
old_data = np.load(EMBED_OLD, allow_pickle=False)
new_data = np.load(EMBED_NEW, allow_pickle=False)
print(f"MiniLM: {old_data['vectors'].shape}")
print(f"SapBERT: {new_data['vectors'].shape}")

# Check if rxcuis match
old_rxcuis = set(old_data["rxcuis"])
new_rxcuis = set(new_data["rxcuis"])
print(f"Common RXCUIs: {len(old_rxcuis & new_rxcuis)}")
print(f"Only in MiniLM: {len(old_rxcuis - new_rxcuis)}")
print(f"Only in SapBERT: {len(new_rxcuis - old_rxcuis)}")
