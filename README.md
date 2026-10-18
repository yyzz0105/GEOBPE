# Protein Geometric Byte Pair Encoding
[![Pip](https://img.shields.io/badge/Pip-blue)](https://pypi.org/project/GeoBPE/0.1.0/)
[![Preprint](https://img.shields.io/badge/Arxiv-green)](https://arxiv.org/abs/2511.11758)
[![OpenReview](https://img.shields.io/badge/OpenReview-red)](https://openreview.net/forum?id=55e5f3GVFc)

![GeoBPE](./data/assets/fig1.png)

This repo contains our implementation of Protein Structure Tokenization via Geometric Byte Pair Encoding (ICLR 2026).

## Overview

Protein structure is central to biological function, and enabling multimodal protein models requires joint reasoning over sequence, structure, and function. We introduce GeoBPE, a geometry-grounded PST that transforms continuous, noisy, multi-scale backbone conformations into discrete ``sentences" of geometry while enforcing global constraints.

Analogous to byte-pair encoding, GeoBPE generates a hierarchical vocabulary of geometric primitives by iteratively (i) clustering Geo-Pair occurrences with k-medoids to yield a resolution-controllable vocabulary; (ii) quantizing each Geo-Pair to its closest medoid prototype; and (iii) reducing drift through differentiable inverse kinematics that optimizes boundary glue angles under an SE(3) end-frame loss.

![GeoBPE](./data/assets/geobpe.jpg)

GeoBPE offers **compression** (>10x reduction in bits-per-residue at similar distortion rate), **data efficiency** (>10x less training data), and **generalization** (maintains test/train distortion ratio of 1.0−1.1). It is architecture-agnostic: (a) its hierarchical vocabulary provides a **strong inductive bias** for coarsening residue-level embeddings from large PLMs into motif- and protein-level representations, **consistently outperforming leading PSTs** across 12 tasks and 24 test splits; (b) paired with a transformer, GeoBPE supports **unconditional backbone generation via language modeling**; and (c) tokens align with CATH functional families and **support expert-interpretable case studies**, offering functional meaning absent in prior PSTs.

## Installation

```bash
conda env create -f environment.yml
conda activate GeoBPE
pip install esm==3.2.0 --no-deps # install ESM
pip install -e ./  # make sure ./ is the root
```

## Reproducibility and Scope

The core protein-tokenizer pipeline is runnable in this repository:

- `bin/encode.py`: extract protein backbone geometry and learn GeoBPE tokens.
- `bin/induce.py`: encode additional structures with a learned tokenizer.
- `bin/train.py`: train the structure language model, with CPU, single-GPU, and `DataParallel` multi-GPU support.
- Sampling/inference: decode token sequences back to 3-bead protein backbone coordinates and write PDB output.

### End-to-end smoke reproduction

The following command runs the complete code path with two training chains and
two validation chains:

```bash
bash scripts/reproduce_geobpe_toy.sh
```

The script performs:

```text
geometry extraction -> 500-bin GeoBPE token learning -> glue IK
-> induce -> one-epoch LM training -> token sampling -> BPE decoding
-> N-CA-C forward-kinematics reconstruction -> PDB output
-> independent token-to-PDB validation -> unit tests
```

Set `PYTHON_BIN` to select a different environment and `RUN_DIR` to select the
output directory. The default output is `runs/reproduce_geobpe_toy`.

The validated local 500-bin smoke artifacts under
`runs/smoke_encode_500_k16` and `runs/smoke_induce_500_k16` show:

- an actual vocabulary size of 1504, including learned motif tokens and 500-bin
  angle/bond-length tokens;
- token-ID round trips of 0.1121 and 0.1215 Angstrom RMSD for the two encoded
  chains;
- reconstruction error against the two original backbones of 4.7373 and 4.0957
  Angstrom backbone RMSD, dominated by the deliberately coarse smoke setup;
- two sampled sequences, each with 257 token IDs, that decode independently to
  65 residues and 195 N/CA/C atoms;
- sampled-token decoding versus independently re-read generated PDBs at 0.0581
  and 0.0541 Angstrom aligned backbone RMSD.

This is a full code-path reproduction, not a numerical reproduction of the
paper benchmarks. The one-epoch/two-chain LM metrics are smoke-test values and
must not be compared with the paper. This checkout contains 72 training PDBs,
37 validation PDBs, 5 CAMEO structures, and 5 CASP14 structures; it does not
contain the complete paper preprocessing result. No official GeoBPE checkpoint
is included or referenced by this repository, so a paper-level inference
comparison cannot be run from the current checkout alone.

The ESM package is installed without its dependency resolver because ESM 3.2.0's published metadata pins an older PyTorch ecosystem (`torchtext` and Biotite) that conflicts with the tested Torch 2.6 environment. The incompatible `torchtext` pin is therefore not installed. The GeoBPE tokenizer, induction, training, and geometric decoding paths work without that metadata being satisfied. The optional PyG wheels were removed from `environment.yml` because the old `pt24` builds conflict with Torch 2.6; only the optional `bin/pdb_to_residues_esm.py` helper needs PyG.

Use `--skip_sctm` when running the language-model pipeline if the protein inverse-folding/SCTM evaluation dependency is unavailable. This skips that optional evaluation branch while leaving tokenization, training, sampling, and geometric reconstruction available.

### What reconstruction actually does

Reconstruction does not read coordinates from `bond_to_token` or from the BPE merge tree. The merge tree records only the token hierarchy. Decoding proceeds by:

1. dequantizing token geometry (`BPE.dequantize`);
2. expanding token geometry with `BPE.recover` and building the decoded structure
   with `BPE.recover_structure`;
3. reconstructing coordinates by forward kinematics (`Tokenizer.compute_coords`, `Tokenizer.geo_nerf`, and `NERFBuilder/place_dihedral`).

The "glue optimization" step updates the three boundary angles `(omega, C-N-CA, phi)` with LBFGS under an SE(3) terminal-frame loss and an empirical angle prior. With `glue-opt-method=all`, all boundaries in one chain are optimized together against all of that chain's cached exit frames; independent chains are parallelized across workers. This is the paper's batch/global boundary-IK refinement, not optimization of side-chain, all-atom, or unconstrained conformational degrees of freedom.

### RNA scope

The current implementation is a protein backbone tokenizer, not an RNA tokenizer. Protein atom names, N-CA-C NERF topology, `ProteinChain`, ESM embeddings, ProteinMPNN/SCTM integrations, PDB output, and structural metrics are hardcoded. Applying GeoBPE to RNA requires an adapter for:

- RNA 3-bead extraction, for example `[P, C4', N1]` for C/U and `[P, C4', N9]` for A/G;
- RNA-specific bond lengths, bond angles, torsions, geometry bins, and token topology;
- an RNA NERF/decoder, sequence model, metrics, and PDB writer.

The conceptual GeoBPE steps transfer to RNA, including local boundary optimization. However, a 3-bead representation is only an intermediate representation for chemically strict reconstruction. Full-atom restoration still requires sugar-phosphate topology, base placement, chirality, ring closure, and phosphodiester connectivity. This is also true for proteins: GeoBPE reconstructs the N-CA-C backbone, not all side-chain atoms. RNA is harder because its sugar ring, phosphate linkage, base orientation, and local flexibility are less constrained by a three-point backbone.

This repository is therefore a reasonably complete reproduction of the core protein pipeline, but it does not claim full reproduction of every paper benchmark, full-atom reconstruction, or direct RNA application without an adapter.

## Downloading data

The RCSB PDB pretraining dataset and CAMEO/CASP test data should be placed under `data/vqvae_pretrain/{train|validation|CAMEO|CASP14}`. For train/validation, we include only a few .pdb files for smoke testing (download the rest through official channels).

The processed task-specific splits are given as .jsonl files under data/struct_token_bench. To download the PDB structures:
```bash
cd data/struct_token_bench

source=InterProFunctionDataset
ds=(binding activesite conservedsite repeat)
for d in "${ds[@]}"; do for f in ${source}_${d}*.jsonl; do python download_pdbs.py --data_file "$f" --output_dir "interpro/$d"; done; done

source=BioLIP2FunctionDataset
ds=(binding catalytic)
for d in "${ds[@]}"; do for f in ${source}_${d}*.jsonl; do python download_pdbs.py --data_file "$f" --output_dir "biolip2/$d"; done; done

source=ProteinGLUEEpitopeRegionDataset
for f in ${source}*.jsonl; do python download_pdbs.py --data_file "$f" --output_dir "proteinglue"; done

source=ProteinShakeBindingSiteDataset
for f in ${source}*.jsonl; do python download_pdbs.py --data_file "$f" --output_dir "proteinshake"; done

source=TapeRemoteHomologyDataset
for f in ${source}*.jsonl; do python download_pdbs.py --data_file "$f" --output_dir "homo"; done
```
For a quick smoke test, use `proteinglue` (ept).

## Run GeoBPE

We include the following resources to make it easy to use GeoBPE:
- **GeoBPE API and Usage Guidelines Doc** ([./docs/hparam_guide.md](./docs/hparam_guide.md)) -- descriptions, intuitions, and guidelines on how to effectively and efficiently use GeoBPE
- **Experiment Logs** ([./docs/GeoBPE-logged-runs.pdf](./docs/GeoBPE-logged-runs.pdf)) -- collection of past experiments varying hyperparameters settings; quickly lookup settings & performance to save future iteration time.

### Learn tokenizer with PDB pretrain train set

```bash
# suggested setting for downstream transfer experiments
./scripts/encode.sh 0 0 3 3 pretrain '1-50' histogram 5 5 false 0 2-2:3-5:5-1:6-2:8-1 500 true 0.0 all true true 10 1000 # slurm ready; prepend sbatch settings if using

# suggested setting for pareto-optimal compression-reconstruction
IK_TRANSLATION_WEIGHT=1.0 ./scripts/encode.sh 0 0 3 3 pretrain '1-500' histogram 5 5 false 0 2-100:3-500:5-20:6-100:8-5:9-20:11-1:12-5:14-1 5000 true 1.0 all true true 10 0
```

The command-line defaults now follow Table 14 of the paper: `p-min-size=0`, `bins={1:500}`, free bond lengths, glue prior `1.0`, `(wR, wt)=(1.0, 0.1)`, `max-num-strucs=5000`, `num-p={2:100,3:500,5:20,6:100}`, and RMSD super-resolution enabled. For the Pareto runs, Table 17/18 use `wt=1.0`; set `IK_TRANSLATION_WEIGHT=1.0` as shown above or pass `--ik-translation-weight 1.0` directly.

Each run creates a folder `ckpts/$d` (e.g. `ckpts/1763070917.6459317`) containing periodic tokenizer checkpoints (`bpe_iter=$iter.pkl`), visualizations, and metrics. You can monitor the running statistics (files matching `*=$iter*`) and stop when desired vocabulary size / token size / segmentation lengths is hit. A suggested stopping iteration (based on LLM scaling heuristic, see [./docs/lm-heuristic.png](./docs/lm-heuristic.png)) is marked in `run_iter=$iter.png`.

### Encode validation set (PDB), test sets (CAMEO, CASP) and analyze the token efficiency.

Once you set `d`
```bash
./scripts/induce.sh 0 0 {prevalid|cameo|casp} ckpts/$d/bpe_post_init.pkl false
```

Each run creates another folder `ckpts/$dval` and the tokenized structures are stored in `ckpts/$dval/$i.pkl`. They are appended to the train set (`ckpts/$d/bpe_post_init.pkl`) and written to `ckpts/$dval/bpe_iter=$iter.pkl`.

### Encode Task-Specific Dataset with Tokenizer at Iteration $iter.

After downloading the structures, run:
```bash
./scripts/induce.sh 0 0 {bindint|bindbio|catint|catbio|conserved|repeat|ept|atlas|bindshake|homo} ckpts/$d/bpe_iter=$iter.pkl false
```

Each run creates another folder `ckpts/$dtask` and the tokenized structures are stored in `ckpts/$dtask/$i.pkl`. They are concatenated and saved in `ckpts/$dtask/bpe_iter=$iter.pkl`.

### Downstream Transfer Prediction Tasks

Tasks 1, 2, ..., 10 correspond to the order above (1 for bindint, ..., 10 for homo). Make sure the dataset is encoded first.
```bash
./scripts/predict.sh 0 {1|2|..|10} ckpts/$dtask/bpe_iter=$iter.pkl
```

### Small Structure Language Model (SSLM) Evaluation

Make sure prevalid is encoded first (command above).

Train a small LM (adapt hyperparams as necessary):
```bash
./scripts/train.sh 0 8 0 ckpts/$dval/bpe_post_init.pkl
```

Pick the latest best ckpt. Sample and compute generative metrics:
```bash
./scripts/train.sh 0 8 0 ckpts/$dval/bpe_post_init.pkl ckpts/$dval/ckpt_epoch${epoch}.pt 1 # this samples 1 backbone, change to as many as you like
```

### Run VQ-VAE baselines (with our SSLM eval loop)

We extend the benchmarking codebase from Yuan et al. (ICML 2025). More details are found on their [public released codebase](https://github.com/KatarinaYuan/StructTokenBench). Follow their steps or run the following to create a env called ``pstbench''.

Pretrain the VQ-VAE with periodic SSLM eval.

```bash
cd StructTokenBench
conda env create -f environment.yml
conda activate pstbench
./pretrain.sh 0 512 0 {vanillavq|aminoaseed}
```

## Citation
If you use GeoBPE in your research, please cite our paper:
```
@inproceedings{sun2025protein,
  title={Protein Structure Tokenization via Geometric Byte Pair Encoding},
  author={Sun, Michael and Yuan, Weize and Liu, Gang and Matusik, Wojciech and Zitnik, Marinka},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2511.11758}
}
```

## Contact

Please contact msun415@mit.edu if you have any questions.
