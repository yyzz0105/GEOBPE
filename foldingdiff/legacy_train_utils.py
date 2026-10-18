"""
Compatibility helpers for the legacy autoregressive FoldingDiff baseline.

The GeoBPE training entry point replaced the original FoldingDiff
``bin/train.py``.  The baseline autoregressive scripts still depend on a few
small helpers from that file, so keep them in the package instead of importing
from a CLI module with different semantics.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import pytorch_lightning as pl


ANGLES_DEFINITIONS = Literal[
    "canonical", "canonical-full-angles", "canonical-minimal-angles", "cart-coords"
]


def build_callbacks(
    outdir: str, early_stop_patience: Optional[int] = None, swa: bool = False
):
    """Build callbacks used by the original FoldingDiff AR trainer."""
    os.makedirs(os.path.join(outdir, "logs/lightning_logs"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "models/best_by_valid"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "models/best_by_train"), exist_ok=True)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(outdir, "models/best_by_valid"),
            save_top_k=5,
            save_weights_only=True,
            mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="train_loss",
            dirpath=os.path.join(outdir, "models/best_by_train"),
            save_top_k=5,
            save_weights_only=True,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]
    if early_stop_patience is not None and early_stop_patience > 0:
        logging.info(f"Using early stopping with patience {early_stop_patience}")
        callbacks.append(
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val_loss",
                patience=early_stop_patience,
                verbose=True,
                mode="min",
            )
        )
    if swa:
        callbacks.append(pl.callbacks.StochasticWeightAveraging())
    logging.info(f"Model callbacks: {callbacks}")
    return callbacks


@pl.utilities.rank_zero_only
def record_args_and_metadata(func_args: Dict[str, Any], results_folder: Path):
    """Write run arguments and the current git revision to the output folder."""
    if results_folder.exists():
        logging.warning(f"Removing old results directory: {results_folder}")
        shutil.rmtree(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    with open(results_folder / "training_args.json", "w") as sink:
        logging.info(f"Writing training args to {sink.name}")
        json.dump(func_args, sink, indent=4)
        for key, value in func_args.items():
            logging.info(f"Training argument: {key}={value}")

    try:
        import git

        repo = git.Repo(
            path=os.path.dirname(os.path.abspath(__file__)),
            search_parent_directories=True,
        )
        sha = repo.head.object.hexsha
        with open(results_folder / "git_sha.txt", "w") as sink:
            sink.write(sha + "\n")
    except git.exc.InvalidGitRepositoryError:
        logging.warning("Could not determine Git repo status -- not a git repo")
    except ModuleNotFoundError:
        logging.warning("Could not determine Git repo status -- GitPython is not installed")
