from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
# from scipy.spatial import cKDTree
from torch_kdtree import build_kd_tree


@dataclass
class KDTree:
    """Thin wrapper over scipy.spatial.cKDTree for nearest neighbor queries."""

    points: torch.Tensor
    device = torch.device('cuda')
    leafsize: int = 16

    def __post_init__(self) -> None:
        if not isinstance(self.points, torch.Tensor):
            self.points = torch.tensor(self.points, device=self.device, dtype=torch.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("KDTree expects (N,3) points")
        # self._tree = cKDTree(self.points, leafsize=self.leafsize, copy_data=True, balanced_tree=True)
        self._tree = build_kd_tree(self.points, device=self.device)

    def query(self, qpoints: Optional[np.ndarray, torch.Tensor], k: int = 1, eps: float = 0.0) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # if not isinstance(qpoints, np.ndarray):
        #     qpoints = np.asarray(qpoints, dtype=np.float64)
        # dists, idxs = self._tree.query(qpoints, k=k, eps=eps)
        if not isinstance(qpoints, torch.Tensor):
            qpoints = torch.tensor(qpoints, device=self.device, dtype=torch.float64)
        dists, idxs = self._tree.query(qpoints, nr_nns_searches=k)
        return dists.reshape(-1), idxs.reshape(-1)

    def query_radius(self, qpoints: np.ndarray, r: float, count_only: bool = False) -> list:
        if not isinstance(qpoints, np.ndarray):
            qpoints = np.asarray(qpoints, dtype=np.float64)
        return self._tree.query_ball_point(qpoints, r, return_sorted=False, return_length=count_only)

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])
