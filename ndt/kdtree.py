from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Union, Sequence

import numpy as np
import torch
from torch_kdtree import build_kd_tree

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]]


@dataclass
class KDTree:
    """
    Thin wrapper around torch_kdtree for nearest-neighbor queries.

    - Accepts (N, 3) points as torch/numpy/sequence.
    - Builds the tree on the chosen device (CUDA if available by default).
    - Returns torch tensors on the same device.
    """
    points: ArrayLike
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: torch.dtype = torch.float64
    # Internal fields
    _points: torch.Tensor = field(init=False, repr=False)
    _tree: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._points = self._to_tensor(self.points, self.device, self.dtype)
        if self._points.ndim != 2 or self._points.shape[1] != 3:
            raise ValueError(f"KDTree expects (N, 3) points, got {tuple(self._points.shape)}")
        if self._points.shape[0] == 0:
            raise ValueError("KDTree received zero points")

        # torch_kdtree expects contiguous float tensor
        self._points = self._points.contiguous()

        # Build tree (no gradients needed)
        with torch.no_grad():
            self._tree = build_kd_tree(self._points, device=self.device)

    # ---------- Public API ----------

    @property
    def n_points(self) -> int:
        return int(self._points.shape[0])

    @property
    def points_tensor(self) -> torch.Tensor:
        """Return the internal (N,3) tensor reference (on self.device)."""
        return self._points

    def query_nearest(self, qpoints: ArrayLike) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1-NN query.

        Args:
            qpoints: (M, 3) query points.

        Returns:
            dists: (M,) L2 distances
            idxs:  (M,) indices into the reference set
        """
        q = self._prep_queries(qpoints)
        dists, idxs = self._tree.query(q, nr_nns_searches=1)  # torch_kdtree returns (M,1)
        # Squeeze the last dimension
        return dists.reshape(-1), idxs.reshape(-1)

    # TODO
    def query_nearest_mahalanobis(self, qpoints: ArrayLike) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1-NN query.

        Args:
            qpoints: (M, 3) query points.

        Returns:
            dists: (M,) L2 distances
            idxs:  (M,) indices into the reference set
        """
        ...

    def query(self, qpoints: ArrayLike, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        k-NN query.

        Args:
            qpoints: (M, 3) query points.
            k: number of neighbors (1..N)

        Returns:
            dists: (M, k) L2 distances
            idxs:  (M, k) indices into the reference set
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if k > self.n_points:
            # Clamp to max available; avoids native errors and is usually what you want in small clouds
            k = self.n_points

        q = self._prep_queries(qpoints)
        dists, idxs = self._tree.query(q, nr_nns_searches=k)
        # Ensure contiguous and correct dtype/device
        return dists.contiguous(), idxs.contiguous()

    # ---------- Helpers ----------

    @staticmethod
    def _to_tensor(x: ArrayLike, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Convert input to a torch tensor on device/dtype without unnecessary copies.
        """
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype, non_blocking=True)
        return torch.as_tensor(x, device=device, dtype=dtype)

    def _prep_queries(self, qpoints: ArrayLike) -> torch.Tensor:
        q = self._to_tensor(qpoints, self.device, self.dtype)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError(f"Query points must be (M, 3), got {tuple(q.shape)}")
        if q.shape[0] == 0:
            raise ValueError("No query points provided")
        return q.contiguous()
