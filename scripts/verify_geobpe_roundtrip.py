#!/usr/bin/env python
"""Audit the GeoBPE token-ID to 3-bead coordinate reconstruction path."""

import argparse
import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from esm.utils.structure.protein_chain import ProteinChain
from foldingdiff.algo import compute_rmsd
from foldingdiff.angles_and_coords import write_coords_to_pdb
from foldingdiff.datasets import extract_backbone_coords


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def verify_chain(bpe, chain_index: int, output_dir: Path) -> dict:
    tokenizer = bpe.tokenizers[chain_index]
    token_ids = bpe.quantize(tokenizer)
    decoded_tokens = bpe.dequantize(token_ids)
    replacement = bpe.recover(decoded_tokens)
    reconstructed = bpe.recover_structure(replacement, decoded_tokens)

    direct_coords = np.asarray(tokenizer.compute_coords(), dtype=float)
    decoded_coords = np.asarray(reconstructed.compute_coords(), dtype=float)
    if direct_coords.shape != decoded_coords.shape:
        raise ValueError(
            f"Coordinate shape mismatch for chain {chain_index}: "
            f"{direct_coords.shape} != {decoded_coords.shape}"
        )

    original_pdb = Path(tokenizer.fname).resolve()
    original_chain = ProteinChain.from_pdb(str(original_pdb))
    decoded_chain = ProteinChain.from_backbone_atom_coordinates(
        decoded_coords.reshape(-1, 3, 3)
    )
    bb_rmsd = float(
        decoded_chain.rmsd(original_chain, only_compute_backbone_rmsd=True)
    )
    lddt = float(np.asarray(decoded_chain.lddt_ca(original_chain)).mean())
    tokenized_geometry_rmsd = float(compute_rmsd(direct_coords, decoded_coords))

    stem = f"chain_{chain_index}_{original_pdb.stem}"
    decoded_pdb = output_dir / f"{stem}_decoded_backbone.pdb"
    write_coords_to_pdb(decoded_coords, str(decoded_pdb))

    token_counts = Counter(str(token_id) for token_id in token_ids)
    hierarchy = [
        {
            "start_bond": int(start),
            "token_id": _json_value(token_id),
            "n_bonds": int(n_bonds),
        }
        for start, token_id, n_bonds in sorted(tokenizer.bond_to_token.values())
    ]
    record = {
        "chain_index": chain_index,
        "original_pdb": str(original_pdb),
        "decoded_pdb": str(decoded_pdb),
        "n_residues": int(tokenizer.n),
        "n_token_ids": int(len(token_ids)),
        "n_tokens": int(len(tokenizer.bond_to_token)),
        "token_ids": [int(token_id) for token_id in token_ids],
        "token_counts": dict(sorted(token_counts.items())),
        "token_hierarchy": hierarchy,
        "decoded_tokens": [_json_value(token) for token in decoded_tokens],
        "tokenized_vs_decoded_backbone_rmsd": tokenized_geometry_rmsd,
        "decoded_vs_original_backbone_rmsd": bb_rmsd,
        "decoded_vs_original_lddt_ca": lddt,
        "decoded_coordinate_shape": list(decoded_coords.shape),
    }
    return record


def _resolve_sampled_pdb_paths(sampled_tokens_path: Path, pdb_dir: Path | None):
    manifest_path = sampled_tokens_path.parent / "sampled_pdb_fnames.json"
    if manifest_path.exists():
        with manifest_path.open() as source:
            manifest = json.load(source)
        return [
            (
                path
                if (path := Path(entry)).is_absolute()
                else sampled_tokens_path.parent / path
            ).resolve()
            for entry in manifest
        ]

    if pdb_dir is None:
        pdb_dir = sampled_tokens_path.parent / "sampled_pdb"
    return sorted(pdb_dir.resolve().glob("generated_*.pdb"))


def verify_sampled_generation(
    bpe,
    sampled_tokens_path: Path,
    pdb_dir: Path | None = None,
) -> dict:
    """Decode sampled IDs independently and compare against written PDBs."""
    sampled_tokens_path = sampled_tokens_path.resolve()
    with sampled_tokens_path.open() as source:
        samples = json.load(source)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{sampled_tokens_path} must contain a non-empty list")

    pdb_paths = _resolve_sampled_pdb_paths(sampled_tokens_path, pdb_dir)
    if len(pdb_paths) != len(samples):
        raise ValueError(
            f"Found {len(samples)} sampled token sequences but {len(pdb_paths)} PDBs"
        )

    records = []
    for sample_index, (sample, pdb_path) in enumerate(zip(samples, pdb_paths)):
        if not pdb_path.exists():
            raise FileNotFoundError(pdb_path)

        decoded_tokens = bpe.dequantize(sample)
        replacement = bpe.recover(decoded_tokens)
        reconstructed = bpe.recover_structure(replacement, decoded_tokens)
        decoded_coords = np.asarray(reconstructed.compute_coords(), dtype=float)

        pdb_coords = extract_backbone_coords(
            str(pdb_path), atoms=["N", "CA", "C"]
        )
        if pdb_coords is None:
            raise ValueError(f"Could not extract N/CA/C coordinates from {pdb_path}")
        pdb_coords = np.asarray(pdb_coords, dtype=float)
        if decoded_coords.shape != pdb_coords.shape:
            raise ValueError(
                f"Coordinate shape mismatch for sample {sample_index}: "
                f"{decoded_coords.shape} != {pdb_coords.shape}"
            )

        coordinate_delta = decoded_coords - pdb_coords
        hierarchy = [
            {
                "start_bond": int(start),
                "token_id": _json_value(token_id),
                "n_bonds": int(n_bonds),
            }
            for start, token_id, n_bonds in sorted(
                reconstructed.bond_to_token.values()
            )
        ]
        records.append(
            {
                "sample_index": sample_index,
                "pdb_path": str(pdb_path),
                "n_sampled_token_ids": int(len(sample)),
                "n_motif_tokens": int(len(reconstructed.bond_to_token)),
                "n_residues": int(reconstructed.n),
                "n_backbone_atoms": int(decoded_coords.shape[0]),
                "all_coordinates_finite": bool(np.isfinite(decoded_coords).all()),
                "aligned_backbone_rmsd_angstrom": float(
                    compute_rmsd(decoded_coords, pdb_coords)
                ),
                "unaligned_coordinate_rmsd_angstrom": float(
                    np.sqrt(np.mean(np.sum(coordinate_delta**2, axis=1)))
                ),
                "max_abs_coordinate_delta_angstrom": float(
                    np.max(np.abs(coordinate_delta))
                ),
                "token_hierarchy": hierarchy,
            }
        )

    return {
        "sampled_tokens": str(sampled_tokens_path),
        "sampled_pdbs": [str(path) for path in pdb_paths],
        "vocab_size": int(bpe.vocab_size),
        "mechanism": (
            "sampled token IDs -> dequantize -> recover angles/distances -> "
            "recover_structure -> compute_coords -> NERF forward kinematics; "
            "the written PDB is independently re-read for comparison"
        ),
        "merge_tree_used_for_coordinate_lookup": False,
        "pdb_writer_may_recenter_and_use_default_unsampled_bond_lengths": True,
        "samples": records,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bpe-pkl",
        type=Path,
        default=Path("runs/smoke_encode_500/bpe_post_init.pkl"),
        help="Learned BPE checkpoint to audit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for JSON reports and reconstructed PDBs.",
    )
    parser.add_argument(
        "--chains",
        type=int,
        nargs="+",
        help="Tokenizer indices to verify. Defaults to the first two chains.",
    )
    parser.add_argument(
        "--sampled-tokens",
        type=Path,
        help=(
            "Also decode this sampled_tokens.json independently and compare "
            "the coordinates with the corresponding written PDBs."
        ),
    )
    parser.add_argument(
        "--sampled-pdb-dir",
        type=Path,
        help="Directory containing generated_*.pdb for --sampled-tokens.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    bpe_path = args.bpe_pkl.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else bpe_path.parent / "roundtrip"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with bpe_path.open("rb") as source:
        bpe = pickle.load(source)

    if args.sampled_tokens:
        sampled_report = verify_sampled_generation(
            bpe,
            sampled_tokens_path=args.sampled_tokens,
            pdb_dir=args.sampled_pdb_dir,
        )
        sampled_report_path = output_dir / "sampled_decode_validation.json"
        with sampled_report_path.open("w") as sink:
            json.dump(
                _json_value(sampled_report), sink, indent=2, allow_nan=True
            )
        logging.info("Wrote sampled-decode report to %s", sampled_report_path)
        print(sampled_report_path)
        return

    chain_indices = args.chains or list(range(min(2, len(bpe.tokenizers))))
    records = []
    for chain_index in chain_indices:
        if not 0 <= chain_index < len(bpe.tokenizers):
            raise IndexError(
                f"chain index {chain_index} outside [0, {len(bpe.tokenizers)})"
            )
        logging.info("Verifying chain index %s", chain_index)
        record = verify_chain(bpe, chain_index, output_dir)
        records.append(record)
        with (output_dir / f"chain_{chain_index}.json").open("w") as sink:
            json.dump(_json_value(record), sink, indent=2, allow_nan=True)

    summary = {
        "bpe_checkpoint": str(bpe_path),
        "vocab_size": int(bpe.vocab_size),
        "mechanism": (
            "token IDs -> dequantize -> recover angles/distances -> "
            "recover_structure -> compute_coords -> NERF forward kinematics"
        ),
        "merge_tree_used_for_coordinate_lookup": False,
        "chains": records,
    }
    summary_path = output_dir / "roundtrip_summary.json"
    with summary_path.open("w") as sink:
        json.dump(_json_value(summary), sink, indent=2, allow_nan=True)
    logging.info("Wrote round-trip report to %s", summary_path)
    print(summary_path)


if __name__ == "__main__":
    main()
