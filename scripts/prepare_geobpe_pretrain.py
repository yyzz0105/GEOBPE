#!/usr/bin/env python
"""Build GeoBPE's per-chain protein PDB pretraining dataset from mmCIF files.

The paper's exact sampled10 chain index is not distributed with this repository.
When that index is available, pass it with ``--cache-json``.  Otherwise this
script can build a deterministic fallback sample from the local mmCIF snapshot.
The provenance and filtering statistics are always recorded in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

try:
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.io.pdbx import convert
    try:
        from biotite.structure.io.pdbx import CIFFile
    except ImportError:
        # Biotite before 0.37 named the text mmCIF reader PDBxFile.
        from biotite.structure.io.pdbx import PDBxFile as CIFFile
except ImportError as exc:  # pragma: no cover - exercised by the CLI
    raise SystemExit(
        "biotite is required. Run this script with the GeoBPE-repro environment."
    ) from exc

try:
    from sklearn.model_selection import train_test_split
except ImportError as exc:  # pragma: no cover - exercised by the CLI
    raise SystemExit("scikit-learn is required for the paper-compatible split") from exc


REQUIRED_BACKBONE = ("N", "CA", "C")
_SAFE_CHAIN_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ChainCandidate:
    pdb_id: str
    chain_id: str
    length: int
    path: str
    split: str = ""

    @property
    def key(self) -> str:
        return f"{self.pdb_id}_{self.chain_id}"

    @property
    def output_name(self) -> str:
        chain = _SAFE_CHAIN_CHARS.sub("_", self.chain_id) or "chain"
        return f"{self.pdb_id.lower()}_{chain}.pdb"


def _read_json(path: Path) -> dict:
    with path.open("rt", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_cif_paths(root: Path) -> list[Path]:
    direct = sorted(root.glob("*.cif"))
    nested = sorted(root.glob("mmcif_files/*.cif"))
    if direct:
        return direct
    if nested:
        return nested
    return sorted(root.rglob("*.cif"))


def _load_cache_candidates(
    cache_json: Path,
    cif_paths: Sequence[Path],
    min_length: int,
    max_length: int,
    workers: int,
) -> list[ChainCandidate]:
    """Use an official sampled chain cache without redefining its membership."""
    cache = _read_json(cache_json)
    if not isinstance(cache, dict):
        raise ValueError(f"{cache_json} must contain a JSON object keyed by PDB_CHAIN")

    path_by_pdb = {path.stem.lower(): path for path in cif_paths}
    requested: dict[str, set[str]] = defaultdict(set)
    skipped: Counter[str] = Counter()
    for raw_key in cache:
        key = str(raw_key).strip()
        if "_" not in key:
            skipped["invalid_cache_key"] += 1
            continue
        pdb_id, chain_id = key.split("_", 1)
        pdb_id = pdb_id.lower()
        if pdb_id not in path_by_pdb:
            skipped["cache_pdb_missing_from_mmcif"] += 1
            continue
        requested[pdb_id].add(chain_id)

    jobs: list[tuple[str, Sequence[str], int, int]] = []
    for pdb_id in sorted(requested):
        path = path_by_pdb[pdb_id]
        jobs.append((str(path), sorted(requested[pdb_id]), min_length, max_length))
    candidates, parse_stats = _run_parse_jobs(jobs, workers)
    skipped.update(parse_stats)

    print(
        f"Official cache mode: requested {len(cache)} chains; "
        f"accepted {len(candidates)}; skipped {dict(skipped)}"
    )
    return candidates


def _nonfinite_coord_mask(residue: struc.AtomArray) -> bool:
    return bool(np.any(~np.isfinite(residue.coord)))


def _parse_cif_file(
    job: tuple[str, Sequence[str] | None, int | None, int | None]
) -> tuple[list[ChainCandidate], Counter[str]]:
    """Parse one mmCIF file and return valid, complete protein-chain candidates."""
    path_string, requested_chains, min_length, max_length = job
    path = Path(path_string)
    stats: Counter[str] = Counter()
    candidates: list[ChainCandidate] = []
    requested = None if requested_chains is None else set(requested_chains)

    try:
        cif = CIFFile.read(str(path))
        atoms = convert.get_structure(
            cif, model=1, altloc="first", extra_fields=["b_factor"]
        )
    except Exception:
        stats["mmcif_parse_error"] += 1
        return candidates, stats

    if atoms.array_length() == 0:
        stats["empty_cif"] += 1
        return candidates, stats

    try:
        protein = atoms[struc.filter_amino_acids(atoms) & ~atoms.hetero]
    except Exception:
        stats["amino_acid_filter_error"] += 1
        return candidates, stats

    if protein.array_length() == 0:
        stats["no_protein_atoms"] += 1
        return candidates, stats

    chain_ids = sorted(str(chain) for chain in np.unique(protein.chain_id))
    for chain_id in chain_ids:
        if requested is not None and chain_id not in requested:
            continue

        chain = protein[protein.chain_id == chain_id]
        residues = list(struc.residue_iter(chain))
        if not residues:
            stats["empty_chain"] += 1
            continue

        length = len(residues)
        if min_length is not None and length < min_length:
            stats["shorter_than_min_length"] += 1
            continue
        if max_length is not None and length > max_length:
            stats["longer_than_max_length"] += 1
            continue

        valid = True
        residue_backbones: list[dict[str, np.ndarray]] = []
        for residue in residues:
            names = set(str(name) for name in residue.atom_name)
            if not all(name in names for name in REQUIRED_BACKBONE):
                stats["incomplete_backbone"] += 1
                valid = False
                break
            if _nonfinite_coord_mask(residue):
                stats["nonfinite_coordinates"] += 1
                valid = False
                break
            residue_backbones.append(
                {
                    str(atom.atom_name): atom.coord
                    for atom in residue
                    if str(atom.atom_name) in REQUIRED_BACKBONE
                }
            )
        if not valid:
            continue

        # GeoBPE's forward kinematics assumes one uninterrupted N-CA-C chain.
        # Reject missing-residue gaps instead of silently joining distant
        # residues when the per-chain PDB is renumbered.
        for previous, current in zip(residue_backbones, residue_backbones[1:]):
            previous_c = previous["C"]
            current_n = current["N"]
            if float(np.linalg.norm(previous_c - current_n)) > 2.0:
                stats["chain_break"] += 1
                valid = False
                break
        if not valid:
            continue

        candidates.append(
            ChainCandidate(
                pdb_id=path.stem.lower(),
                chain_id=chain_id,
                length=length,
                path=str(path),
            )
        )

    if requested is not None:
        found = {candidate.chain_id for candidate in candidates}
        stats["requested_chain_not_found_or_invalid"] += len(requested - found)

    return candidates, stats


def _run_parse_jobs(
    jobs: Iterable[tuple[str, Sequence[str] | None, int | None, int | None]],
    workers: int,
) -> tuple[list[ChainCandidate], Counter[str]]:
    candidates: list[ChainCandidate] = []
    stats: Counter[str] = Counter()
    jobs = list(jobs)
    if workers <= 1:
        results = map(_parse_cif_file, jobs)
    else:
        pool = mp.Pool(processes=workers)
        try:
            results = pool.imap_unordered(_parse_cif_file, jobs, chunksize=8)
            for file_candidates, file_stats in results:
                candidates.extend(file_candidates)
                stats.update(file_stats)
        finally:
            pool.close()
            pool.join()
        return candidates, stats

    for file_candidates, file_stats in results:
        candidates.extend(file_candidates)
        stats.update(file_stats)
    return candidates, stats


def _discover_candidates(
    cif_paths: Sequence[Path], min_length: int, max_length: int, workers: int
) -> tuple[list[ChainCandidate], Counter[str]]:
    jobs = ((str(path), None, min_length, max_length) for path in cif_paths)
    return _run_parse_jobs(jobs, workers)


def _fallback_sample(
    candidates: Sequence[ChainCandidate], sample_size: int, seed: int
) -> list[ChainCandidate]:
    ordered = sorted(candidates, key=lambda candidate: candidate.key.upper())
    if len(ordered) <= sample_size:
        return ordered
    _, selected = train_test_split(
        ordered,
        train_size=sample_size,
        random_state=seed,
        shuffle=True,
    )
    return sorted(selected, key=lambda candidate: candidate.key.upper())


def _write_chain(candidate: ChainCandidate, split: str, output_root: Path) -> Counter:
    stats: Counter[str] = Counter()
    path = Path(candidate.path)
    try:
        atoms = convert.get_structure(
            CIFFile.read(str(path)), model=1, altloc="first", extra_fields=["b_factor"]
        )
        chain = atoms[
            struc.filter_amino_acids(atoms)
            & ~atoms.hetero
            & (atoms.chain_id == candidate.chain_id)
        ]
    except Exception:
        stats["write_parse_error"] += 1
        return stats

    if chain.array_length() == 0:
        stats["write_empty_chain"] += 1
        return stats

    # The output contains one chain per file. Renumbering makes the PDB safe for
    # GeoBPE's strict monotonic residue-index check while preserving atom order.
    chain = chain.copy()
    next_res_id = 1
    previous_key = None
    for index, atom in enumerate(chain):
        key = (
            int(atom.res_id),
            str(getattr(atom, "ins_code", "")),
        )
        if previous_key is None or key != previous_key:
            if previous_key is not None:
                next_res_id += 1
            previous_key = key
        chain.res_id[index] = next_res_id
    chain.chain_id[:] = "A"
    if "ins_code" in chain.get_annotation_categories():
        chain.ins_code[:] = ""

    output_path = output_root / split / candidate.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".pdb.tmp")
    try:
        pdb_file = PDBFile()
        pdb_file.set_structure(chain)
        pdb_file.write(str(temporary_path))
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        stats["pdb_write_error"] += 1
        return stats

    stats["written"] += 1
    return stats


def _write_chains(
    candidates: Sequence[ChainCandidate],
    output_root: Path,
    workers: int,
    overwrite: bool,
) -> Counter:
    stats: Counter[str] = Counter()
    pending: list[tuple[ChainCandidate, str, Path]] = []
    for candidate in candidates:
        split = "train" if candidate.split == "train" else "validation"
        output_path = output_root / split / candidate.output_name
        if output_path.exists() and not overwrite:
            stats["already_exists"] += 1
            continue
        pending.append((candidate, split, output_root))

    if not pending:
        return stats

    if workers <= 1:
        results = map(lambda item: _write_chain(*item), pending)
    else:
        pool = mp.Pool(processes=workers)
        results = pool.starmap(_write_chain, pending, chunksize=8)
        pool.close()
        pool.join()

    for result in results:
        stats.update(result)
    return stats


def _attach_split(
    candidates: Sequence[ChainCandidate], test_size: float, seed: int
) -> list[ChainCandidate]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    indices = list(range(len(candidates)))
    train_indices, _ = train_test_split(
        indices, test_size=test_size, random_state=seed, shuffle=True
    )
    train_set = set(train_indices)
    return [
        ChainCandidate(
            pdb_id=candidate.pdb_id,
            chain_id=candidate.chain_id,
            length=candidate.length,
            path=candidate.path,
            split="train" if index in train_set else "validation",
        )
        for index, candidate in enumerate(candidates)
    ]


def _chain_to_dict(candidate: ChainCandidate) -> dict:
    return {
        "key": candidate.key,
        "pdb_id": candidate.pdb_id,
        "chain_id": candidate.chain_id,
        "length": candidate.length,
        "split": candidate.split,
        "output_name": candidate.output_name,
    }


def _length_summary(candidates: Sequence[ChainCandidate]) -> dict:
    lengths = np.asarray([candidate.length for candidate in candidates], dtype=np.int64)
    if lengths.size == 0:
        return {"count": 0}
    return {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmcif-dir",
        type=Path,
        default=Path("data/pdb_data/mmcif_files"),
    )
    parser.add_argument(
        "--cache-json",
        type=Path,
        default=None,
        help="Official filtered_chain_data_cache-sampled10.json, if available",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/vqvae_pretrain"),
    )
    parser.add_argument("--min-length", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--fallback-sample-size", type=int, default=48316)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0, help="Limit files for smoke tests")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    args.mmcif_dir = args.mmcif_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.mmcif_dir.is_dir():
        raise SystemExit(f"mmCIF directory does not exist: {args.mmcif_dir}")

    cif_paths = _iter_cif_paths(args.mmcif_dir)
    if not cif_paths:
        raise SystemExit(f"No .cif files found below {args.mmcif_dir}")
    if args.limit:
        cif_paths = cif_paths[: args.limit]

    mode = "official-cache"
    if args.cache_json is not None:
        cache_path = args.cache_json.resolve()
        if not cache_path.is_file():
            raise SystemExit(f"Cache JSON does not exist: {cache_path}")
        candidates = _load_cache_candidates(
            cache_path,
            cif_paths,
            args.min_length,
            args.max_length,
            args.workers,
        )
    else:
        mode = "deterministic-fallback"
        candidates, discovery_stats = _discover_candidates(
            cif_paths, args.min_length, args.max_length, args.workers
        )
        print(
            f"Discovered {len(candidates)} valid chains before fallback sampling; "
            f"skip stats: {dict(discovery_stats)}"
        )
        candidates = _fallback_sample(
            candidates, args.fallback_sample_size, args.seed
        )

    candidates = sorted(candidates, key=lambda candidate: candidate.key.upper())
    if not candidates:
        raise SystemExit("No valid protein chains passed filtering; nothing to split")
    candidates = _attach_split(candidates, args.test_size, args.seed)
    train_count = sum(candidate.split == "train" for candidate in candidates)
    validation_count = len(candidates) - train_count

    manifest = {
        "mode": mode,
        "cache_json": str(args.cache_json.resolve()) if args.cache_json else None,
        "mmcif_dir": str(args.mmcif_dir),
        "mmcif_files_scanned": len(cif_paths),
        "min_length": args.min_length,
        "max_length": args.max_length,
        "fallback_sample_size": (
            args.fallback_sample_size if mode == "deterministic-fallback" else None
        ),
        "seed": args.seed,
        "test_size": args.test_size,
        "selected_chains": len(candidates),
        "train_chains": train_count,
        "validation_chains": validation_count,
        "length_summary": _length_summary(candidates),
        "paper_exact_chain_index": mode == "official-cache",
        "paper_exact_split": False,
        "split_note": (
            "The official cache controls chain membership only. Train/validation "
            "assignment is generated here with the recorded seed."
        ),
        "chains": [_chain_to_dict(candidate) for candidate in candidates],
        "elapsed_seconds_before_write": round(time.time() - started, 3),
    }

    if args.dry_run:
        dry_manifest = args.output_dir / "prepare_dry_run_manifest.json"
        dry_manifest.parent.mkdir(parents=True, exist_ok=True)
        dry_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(
            f"Dry run: selected {len(candidates)} chains "
            f"({train_count} train, {validation_count} validation). "
            f"Manifest: {dry_manifest}"
        )
        return 0

    if args.overwrite:
        for split in ("train", "validation"):
            split_dir = args.output_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

    write_stats = _write_chains(
        candidates, args.output_dir, args.workers, overwrite=args.overwrite
    )
    manifest["write_stats"] = dict(write_stats)
    manifest["elapsed_seconds"] = round(time.time() - started, 3)
    manifest_path = args.output_dir / "prepare_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote dataset to {args.output_dir}: {train_count} train, "
        f"{validation_count} validation. Stats: {dict(write_stats)}"
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
