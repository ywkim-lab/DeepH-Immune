#!/usr/bin/env python3
"""Regenerate frozen ESM-2 mean-pooled peptide and MHC embeddings.

This step is optional because the repository already contains the precomputed
features needed for training and external validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--name", required=True, help="Filename suffix, e.g. training_Immuscope_7092")
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


@torch.inference_mode()
def embed_unique(sequences, tokenizer, model, device, batch_size):
    unique = list(dict.fromkeys(map(str, sequences)))
    result = {}
    for start in tqdm(range(0, len(unique), batch_size), desc="ESM-2 unique sequences"):
        seqs = unique[start:start+batch_size]
        enc = tokenizer(
            seqs, return_tensors="pt", padding=True,
            return_special_tokens_mask=True,
        )
        special = enc.pop("special_tokens_mask").to(device)
        enc = {k: v.to(device) for k,v in enc.items()}
        hidden = model(**enc).last_hidden_state

        # Pool real residues only. CLS/EOS and batch padding must not contribute.
        residue_mask = (enc["attention_mask"] > 0) & (special == 0)
        denom = residue_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (hidden * residue_mask.unsqueeze(-1)).sum(dim=1) / denom
        for seq, vec in zip(seqs, pooled.cpu().numpy().astype(np.float32)):
            result[seq] = vec
    return result


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv, low_memory=False)
    required = {"sample_id", "Peptide", "MHC_ungapped", "Immunogenicity"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmModel.from_pretrained(args.model, add_pooling_layer=False).eval().to(device)

    pep_map = embed_unique(df["Peptide"], tokenizer, model, device, args.batch_size)
    mhc_map = embed_unique(df["MHC_ungapped"], tokenizer, model, device, args.batch_size)
    pep = np.stack([pep_map[str(s)] for s in df["Peptide"]])
    mhc = np.stack([mhc_map[str(s)] for s in df["MHC_ungapped"]])
    labels = df["Immunogenicity"].to_numpy(dtype=np.int32)
    ids = df["sample_id"].astype(str).to_numpy()

    np.save(args.output_dir / f"ESM2_peptide_{args.name}.npy", pep)
    np.save(args.output_dir / f"ESM2_mhc_{args.name}.npy", mhc)
    np.save(args.output_dir / f"Labels_{args.name}.npy", labels)
    np.save(args.output_dir / f"Sample_ids_{args.name}.npy", ids)
    pd.DataFrame({"sample_id": ids, "row_index": np.arange(len(df))}).to_csv(
        args.output_dir / f"ESM2_row_manifest_{args.name}.csv", index=False
    )
    print("Saved:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
