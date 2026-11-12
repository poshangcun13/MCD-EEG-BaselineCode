"""Shared paths for the supplementary MCD-EEG experiments.

Python dependencies: numpy, scipy, pandas, scikit-learn, torch, vit-pytorch,
requests. Training uses a CUDA device with BF16 support.

Set MCD_EEG_DATA_ROOT to the downloaded dataset root. Expected subdirectories:
  阅读阶段/EEG_{deaf,hearing}.mat
  记忆阶段/EEG_{deaf,hearing}_memory.mat
  IQ/EEG_deaf_IQ.mat and IQ/EEG_hearing_IQ.npz (or a valid hearing MAT)
  N-back/静息态/resting_{first,rewear}_10subjects.mat
  N-back/N-back/{0,1,2}-BACK-{1,2}.mat

Optional environment variables: MCD_EEG_OUTPUT_ROOT, MCD_EEG_WEIGHTS_ROOT.
Data, outputs, and weights are not distributed with this code.

Entry points:
  python run_resting_benchmark.py       # Table 6, resting state
  python run_nback_benchmark.py         # Table 6, matched N-back loads
  python run_cognitive_benchmark.py     # Table 8, cognitive-stage decoding
  python download_checkpoint.py        # Public EEGPT checkpoint, verified hash
  python prepare_weights.py            # Extract the frozen target encoder
  python run_pretrained_benchmark.py   # Table 11, six transfer conditions
  python run_factorial_nback.py        # Table 9, load x recording controls
  python summarize_results.py          # Identity-result mean/sample SD

The paths below can also be edited to match a different dataset layout.
"""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("MCD_EEG_DATA_ROOT", CODE_ROOT / "data")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("MCD_EEG_OUTPUT_ROOT", CODE_ROOT / "outputs")).expanduser().resolve()
WEIGHTS_ROOT = Path(os.environ.get("MCD_EEG_WEIGHTS_ROOT", CODE_ROOT / "weights")).expanduser().resolve()
NB_DATA = DATA_ROOT / "N-back" / "N-back"
REST_DATA = DATA_ROOT / "N-back" / "静息态"
