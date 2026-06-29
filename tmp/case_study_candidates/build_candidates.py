"""Build a structured candidate table for the BioSNAP case-study match.

Loads the UniProt JSONs already fetched into this directory and writes
`candidates.json` with: drug_name, drug_smiles_raw, drug_inchikey, target_uniprot,
target_name, target_sequence, binding_residues, active_residues, pdb_refs.

We embed the drug SMILES (from DrugBank/PubChem) here and rely on RDKit to
canonicalize on both sides during matching, so encoding differences do not
break the match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdkit import Chem


CANDIDATES = [
    {
        "drug_name": "Imatinib",
        "drug_drugbank": "DB00619",
        "drug_smiles_raw": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "target_uniprot": "P00519",
        "target_label": "ABL1",
        "expected_label": 1,
        "pdb_refs": ["1IEP", "2HYY"],
    },
    {
        "drug_name": "Methotrexate",
        "drug_drugbank": "DB00563",
        "drug_smiles_raw": "CN(CC1=CN=C2N=C(N)N=C(N)C2=N1)C1=CC=C(C=C1)C(=O)N[C@@H](CCC(O)=O)C(O)=O",
        "target_uniprot": "P00374",
        "target_label": "DHFR",
        "expected_label": 1,
        "pdb_refs": ["1U72"],
    },
    {
        "drug_name": "Gefitinib",
        "drug_drugbank": "DB00317",
        "drug_smiles_raw": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
        "target_uniprot": "P00533",
        "target_label": "EGFR",
        "expected_label": 1,
        "pdb_refs": ["2ITY"],
    },
    {
        "drug_name": "Erlotinib",
        "drug_drugbank": "DB00530",
        "drug_smiles_raw": "COCCOC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC(=C3)C#C)OCCOC",
        "target_uniprot": "P00533",
        "target_label": "EGFR",
        "expected_label": 1,
        "pdb_refs": ["1M17"],
    },
    {
        "drug_name": "Dasatinib",
        "drug_drugbank": "DB01254",
        "drug_smiles_raw": "CC1=C(C(=CC=C1)Cl)NC(=O)C2=CN=C(S2)NC3=CC(=NC(=N3)C)N4CCN(CC4)CCO",
        "target_uniprot": "P00519",
        "target_label": "ABL1",
        "expected_label": 1,
        "pdb_refs": ["2GQG"],
    },
    {
        "drug_name": "Dasatinib",
        "drug_drugbank": "DB01254",
        "drug_smiles_raw": "CC1=C(C(=CC=C1)Cl)NC(=O)C2=CN=C(S2)NC3=CC(=NC(=N3)C)N4CCN(CC4)CCO",
        "target_uniprot": "P12931",
        "target_label": "SRC",
        "expected_label": 1,
        "pdb_refs": ["3G5D"],
    },
    {
        "drug_name": "Aspirin",
        "drug_drugbank": "DB00945",
        "drug_smiles_raw": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "target_uniprot": "P23219",
        "target_label": "COX1 (PTGS1)",
        "expected_label": 1,
        "pdb_refs": ["1PTH"],
    },
    {
        "drug_name": "Celecoxib",
        "drug_drugbank": "DB00482",
        "drug_smiles_raw": "CC1=CC=C(C=C1)C1=CC(C(F)(F)F)=NN1C1=CC=C(C=C1)S(N)(=O)=O",
        "target_uniprot": "P35354",
        "target_label": "COX2 (PTGS2)",
        "expected_label": 1,
        "pdb_refs": ["3LN1"],
    },
    {
        "drug_name": "Tamoxifen",
        "drug_drugbank": "DB00675",
        "drug_smiles_raw": "CC/C(=C(/C1=CC=CC=C1)\\C2=CC=C(C=C2)OCCN(C)C)/C3=CC=CC=C3",
        "target_uniprot": "P03372",
        "target_label": "ESR1",
        "expected_label": 1,
        "pdb_refs": ["3ERT"],
    },
    {
        "drug_name": "Fluoxetine",
        "drug_drugbank": "DB00472",
        "drug_smiles_raw": "CNCCC(c1ccccc1)Oc2ccc(cc2)C(F)(F)F",
        "target_uniprot": "P31645",
        "target_label": "SLC6A4 (SERT)",
        "expected_label": 1,
        "pdb_refs": ["6AWO"],
    },
    {
        "drug_name": "Sildenafil",
        "drug_drugbank": "DB00203",
        "drug_smiles_raw": "CCCC1=NN(C)C2=C1NC(=NC2=O)C1=CC(=CC=C1OCC)S(=O)(=O)N1CCN(C)CC1",
        "target_uniprot": "O76074",
        "target_label": "PDE5A",
        "expected_label": 1,
        "pdb_refs": ["2H42"],
    },
]


def canonical_smiles(s: str) -> str:
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def inchikey(s: str) -> str:
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToInchiKey(mol) if mol is not None else ""


def extract_features(uni: dict[str, Any]) -> dict[str, list]:
    binding, active = [], []
    for ft in uni.get("features", []):
        loc = ft.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        if start is None or end is None:
            continue
        residues = list(range(int(start), int(end) + 1))
        desc = ft.get("description", "") or ft.get("type", "")
        ligand = (ft.get("ligand") or {}).get("name", "")
        item = {"start": int(start), "end": int(end), "residues": residues,
                "description": desc, "ligand": ligand}
        ftype = ft.get("type", "")
        if ftype == "Binding site":
            binding.append(item)
        elif ftype == "Active site":
            active.append(item)
    return {"binding": binding, "active": active}


def main() -> None:
    here = Path(__file__).resolve().parent
    out_rows = []
    for cand in CANDIDATES:
        uni_path = here / f"{cand['target_uniprot']}.json"
        if not uni_path.exists():
            print(f"[skip] missing UniProt JSON for {cand['target_uniprot']}")
            continue
        uni = json.loads(uni_path.read_text())
        seq = uni.get("sequence", {}).get("value", "")
        feats = extract_features(uni)
        canon = canonical_smiles(cand["drug_smiles_raw"])
        inchi = inchikey(cand["drug_smiles_raw"])
        if not canon:
            print(f"[skip] invalid SMILES for {cand['drug_name']}")
            continue
        rec = {
            "drug_name": cand["drug_name"],
            "drug_drugbank": cand["drug_drugbank"],
            "drug_smiles_raw": cand["drug_smiles_raw"],
            "drug_smiles_canonical": canon,
            "drug_inchikey": inchi,
            "target_uniprot": cand["target_uniprot"],
            "target_label": cand["target_label"],
            "target_sequence": seq,
            "target_length": len(seq),
            "binding_sites": feats["binding"],
            "active_sites": feats["active"],
            "pdb_refs": cand["pdb_refs"],
            "expected_label": cand["expected_label"],
        }
        out_rows.append(rec)
        bres = sum(len(b["residues"]) for b in feats["binding"])
        ares = sum(len(a["residues"]) for a in feats["active"])
        print(f"  {cand['drug_name']:14s} + {cand['target_label']:18s}  L={len(seq):4d}  binding-res={bres:3d}  active-res={ares}")

    out_path = here / "candidates.json"
    out_path.write_text(json.dumps(out_rows, indent=2))
    print(f"\nWrote {len(out_rows)} candidates → {out_path}")


if __name__ == "__main__":
    main()
