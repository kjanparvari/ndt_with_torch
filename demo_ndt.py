from __future__ import annotations

import os

SEED = 27

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random

import numpy as np
import torch

from ndt import NDTModel


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    os.environ["XDG_SESSION_TYPE"] = "x11"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    seed_everything()

    # Build NDT model on target
    model = NDTModel()
    model.gen.manual_seed(SEED)
    model.build_gaussian_map()

    model.evaluate(log_tb=True)



if __name__ == "__main__":
    main()
