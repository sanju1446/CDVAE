"""
train_dataset.py
================
Reads .pt files built by build_dataset.py from MP20 CSV.
Guarantees proper shapes for all fields including energy labels.
"""

import os
import torch
from torch.utils.data import Dataset


class CrystalGraphDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir    = data_dir
        self.batch_files = sorted(
            [os.path.join(data_dir, f)
             for f in os.listdir(data_dir) if f.endswith(".pt")]
        )

        self.index_map = []
        print("Building index map...")
        for file_id, path in enumerate(self.batch_files):
            batch = torch.load(path, map_location="cpu", weights_only=False)
            for graph_id in range(len(batch)):
                self.index_map.append((file_id, graph_id))
        print(f"Total graphs: {len(self.index_map)}")

        self._cache      = {}
        self._cache_size = 10

    def _load_file(self, file_id):
        if file_id not in self._cache:
            if len(self._cache) >= self._cache_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[file_id] = torch.load(
                self.batch_files[file_id],
                map_location="cpu",
                weights_only=False,
            )
        return self._cache[file_id]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_id, graph_id = self.index_map[idx]
        batch = self._load_file(file_id)
        graph = batch[graph_id]

        N = graph.x.shape[0]

        # ── Atom types: long, valid range ─────────────────────────────────────
        graph.x = graph.x.long().view(-1).clamp(1, 118)

        # ── Edge index: long ──────────────────────────────────────────────────
        graph.edge_index = graph.edge_index.long()

        # ── Lattice: always (1, 9) so DataLoader batches to (B, 9) ───────────
        if hasattr(graph, "lattice") and graph.lattice is not None:
            graph.lattice = graph.lattice.float().view(1, 9)
        else:
            graph.lattice = torch.eye(3).view(1, 9)

        # ── Fractional coords: always (N, 3) in [0, 1) ───────────────────────
        if hasattr(graph, "pos") and graph.pos is not None:
            pos = graph.pos.float().view(N, 3)
            pos = pos % 1.0                        # enforce [0, 1)
            graph.pos = pos
        else:
            graph.pos = torch.zeros(N, 3)

        # ── Edge attributes: always (E,) distances in Angstroms ───────────────
        E = graph.edge_index.shape[1]
        if hasattr(graph, "edge_attr") and graph.edge_attr is not None \
                and graph.edge_attr.numel() == E:
            graph.edge_attr = graph.edge_attr.float().view(-1).clamp(min=1e-6)
        else:
            if E > 0:
                src, dst = graph.edge_index[0], graph.edge_index[1]
                diff     = graph.pos[src] - graph.pos[dst]
                L        = graph.lattice.view(3, 3)
                diff_c   = diff @ L
                graph.edge_attr = diff_c.norm(dim=-1).clamp(min=1e-6)
            else:
                graph.edge_attr = torch.zeros(0)

        # ── Formation energy: (1,) ────────────────────────────────────────────
        if hasattr(graph, "y") and graph.y is not None:
            graph.y = graph.y.float().view(1)
        else:
            graph.y = torch.tensor([float("nan")])

        # ── E above hull: (1,) ────────────────────────────────────────────────
        if hasattr(graph, "e_above_hull") and graph.e_above_hull is not None:
            graph.e_above_hull = graph.e_above_hull.float().view(1)
        else:
            graph.e_above_hull = torch.tensor([float("nan")])

        # ── Band gap: (1,) ────────────────────────────────────────────────────
        if hasattr(graph, "band_gap") and graph.band_gap is not None:
            graph.band_gap = graph.band_gap.float().view(1)
        else:
            graph.band_gap = torch.tensor([float("nan")])

        return graph