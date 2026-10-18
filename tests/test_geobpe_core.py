import os
import pickle
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from foldingdiff.algo import compute_rmsd
from foldingdiff.bpe import BPE, _effective_cpus
from foldingdiff.tokenizer import Tokenizer
from bin.train import SeqDataset, _joint_vocab_ranges


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_BPE = PROJECT_ROOT / "runs/smoke_encode_500/bpe_post_init.pkl"


def make_test_tokenizer(n_residues=4):
    structure = Tokenizer.init_structure(n_residues)
    angles = structure["angles"]
    angles["0C:1N"] = [1.34] * n_residues
    angles["N:CA"] = [1.46] * n_residues
    angles["CA:C"] = [1.54] * n_residues
    angles["phi"] = [-1.2, -0.8, 0.4, 1.1]
    angles["psi"] = [1.0, 0.5, -0.7, -1.1]
    angles["omega"] = [np.pi - 0.05, -np.pi + 0.08, np.pi - 0.03, -np.pi + 0.06]
    angles["tau"] = [1.95, 2.0, 1.9, 2.02]
    angles["CA:C:1N"] = [2.05, 2.1, 1.98, 2.03]
    angles["C:1N:1CA"] = [2.1, 2.0, 2.08, 1.96]
    return Tokenizer(structure)


class GeometryTests(unittest.TestCase):
    def test_geometry_nerf_produces_3n_coordinates(self):
        tokenizer = make_test_tokenizer(4)
        coords = tokenizer.compute_coords()
        self.assertEqual(coords.shape, (12, 3))
        self.assertTrue(np.isfinite(coords).all())

    def test_tokenize_emits_motif_theta_omega_phi_order(self):
        tokenizer = make_test_tokenizer(4)
        tokenizer.bond_to_token = {
            0: (0, 0, 3),
            3: (3, 0, 3),
            6: (6, 0, 3),
            9: (9, 0, 2),
        }

        tokens = tokenizer.tokenize()
        token_types = [(token[0], token[1]) for token in tokens]
        expected_cycle = [
            ("MOTIF", 0),
            ("BOND_ANGLE", "C:1N:1CA"),
            ("DIHEDRAL", "omega"),
            ("DIHEDRAL", "phi"),
        ]
        self.assertEqual(token_types, expected_cycle * 3 + [("MOTIF", 0)])

    @unittest.skipUnless(SMOKE_BPE.exists(), "smoke BPE checkpoint is unavailable")
    def test_token_id_decode_preserves_geometry(self):
        with SMOKE_BPE.open("rb") as source:
            bpe = pickle.load(source)
        tokenizer = bpe.tokenizers[0]

        token_ids = bpe.quantize(tokenizer)
        self.assertTrue(all(0 <= token_id < bpe.vocab_size for token_id in token_ids))
        decoded = bpe.dequantize(token_ids)
        recovered = bpe.recover(decoded)
        reconstructed = bpe.recover_structure(recovered, decoded)

        expected = tokenizer.compute_coords()
        actual = reconstructed.compute_coords()
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(np.isfinite(actual).all())
        self.assertLess(compute_rmsd(expected, actual), 0.5)

    def test_so3_geodesic_loss_matches_small_rotation(self):
        angle = 0.3
        rotation = torch.tensor(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        loss = BPE.so3_geodesic_loss(torch.eye(3), rotation)
        self.assertAlmostEqual(float(loss), angle**2, places=5)


class ExecutionPathTests(unittest.TestCase):
    def test_lm_labels_keep_token_zero_and_mask_padding(self):
        dataset = SeqDataset([[0, 5, 0]], max_len=5)
        item = dataset[0]
        self.assertEqual(item["input_ids"].tolist(), [0, 5, 0, 0, 0])
        self.assertEqual(item["labels"].tolist(), [5, 0, -100, -100, -100])

    def test_zero_slurm_cpu_count_selects_serial_path(self):
        with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "0"}):
            self.assertEqual(_effective_cpus(), 0)

    @unittest.skipUnless(SMOKE_BPE.exists(), "smoke BPE checkpoint is unavailable")
    def test_joint_vocab_ranges_follow_paper_order(self):
        with SMOKE_BPE.open("rb") as source:
            bpe = pickle.load(source)

        ranges = _joint_vocab_ranges(bpe)
        decoded = bpe.dequantize(list(range(bpe.vocab_size)))
        expected = {
            0: ("MOTIF", None),
            1: ("BOND_ANGLE", "C:1N:1CA"),
            2: ("DIHEDRAL", "omega"),
            3: ("DIHEDRAL", "phi"),
        }
        for step, (prefix, key) in expected.items():
            lo, hi = ranges[step]
            for token in decoded[lo:hi]:
                self.assertEqual(token[0], prefix)
                if key is not None:
                    self.assertEqual(token[1], key)

    @unittest.skipUnless(SMOKE_BPE.exists(), "smoke BPE checkpoint is unavailable")
    def test_global_glue_ik_worker(self):
        with SMOKE_BPE.open("rb") as source:
            bpe = pickle.load(source)
        tokenizer = bpe.tokenizers[0]
        if isinstance(tokenizer.cached_all_frames, (str, os.PathLike)):
            tokenizer.cached_all_frames = str(
                (PROJECT_ROOT / tokenizer.cached_all_frames).resolve()
            )

        BPE._init_opt_glue_worker(
            bpe._bin_centers,
            bpe._bin_weights,
            bpe._thresholds,
            bpe.glue_opt_prior,
            bpe.ik_rotation_weight,
            bpe.ik_translation_weight,
        )
        optimized = BPE._opt_glue_worker(tokenizer)
        coords = optimized.compute_coords()
        self.assertEqual(coords.shape, (3 * optimized.n, 3))
        self.assertTrue(np.isfinite(coords).all())


if __name__ == "__main__":
    unittest.main()
