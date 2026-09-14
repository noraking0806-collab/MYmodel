"""Sparse region-graph consistency for binary unsupervised segmentation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SparseRegionGraphConsistency(nn.Module):
    """Regularize a small region graph with attraction and boundary repulsion.

    The graph is built from detached EMA region codes. Gradients flow through
    online region predictions only, while detached cluster centres prevent the
    graph term and the unsupervised probe from forming a trivial co-adaptation
    shortcut.
    """

    def __init__(
        self,
        grid_size: int = 6,
        neighbors: int = 4,
        repulsion_weight: float = 0.5,
        negative_quantile: float = 0.25,
        prediction_temperature: float = 0.2,
        negative_margin: float = 0.25,
        attraction_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if grid_size <= 1:
            raise ValueError("grid_size must be greater than one")
        if neighbors <= 0:
            raise ValueError("neighbors must be positive")
        if repulsion_weight < 0.0:
            raise ValueError("repulsion_weight must be non-negative")
        if attraction_weight < 0.0:
            raise ValueError("attraction_weight must be non-negative")
        if not 0.0 < negative_quantile <= 1.0:
            raise ValueError("negative_quantile must be in (0, 1]")
        if prediction_temperature <= 0.0:
            raise ValueError("prediction_temperature must be positive")
        if not 0.0 <= negative_margin <= 1.0:
            raise ValueError("negative_margin must be in [0, 1]")

        self.grid_size = int(grid_size)
        self.neighbors = int(neighbors)
        self.repulsion_weight = float(repulsion_weight)
        self.negative_quantile = float(negative_quantile)
        self.prediction_temperature = float(prediction_temperature)
        self.negative_margin = float(negative_margin)
        self.attraction_weight = float(attraction_weight)

    @staticmethod
    def _grid_edges(
        grid: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizontal_source: list[int] = []
        horizontal_target: list[int] = []
        vertical_source: list[int] = []
        vertical_target: list[int] = []
        for row in range(grid):
            for column in range(grid - 1):
                source = row * grid + column
                horizontal_source.append(source)
                horizontal_target.append(source + 1)
        for row in range(grid - 1):
            for column in range(grid):
                source = row * grid + column
                vertical_source.append(source)
                vertical_target.append(source + grid)
        source = torch.tensor(
            horizontal_source + vertical_source,
            dtype=torch.long,
            device=device,
        )
        target = torch.tensor(
            horizontal_target + vertical_target,
            dtype=torch.long,
            device=device,
        )
        return source, target

    def forward(
        self,
        online_code: torch.Tensor,
        ema_code: torch.Tensor,
        cluster_centres: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if online_code.shape != ema_code.shape or online_code.ndim != 4:
            raise ValueError("Online and EMA codes must be matching BCHW tensors")
        if (
            cluster_centres.ndim != 2
            or cluster_centres.shape[1] != online_code.shape[1]
        ):
            raise ValueError("Cluster centres do not match the code dimension")

        grid = min(self.grid_size, online_code.shape[-2], online_code.shape[-1])
        online_nodes = F.adaptive_avg_pool2d(online_code, (grid, grid))
        online_nodes = online_nodes.flatten(2).transpose(1, 2).float()
        with torch.no_grad():
            teacher_nodes = F.adaptive_avg_pool2d(ema_code, (grid, grid))
            teacher_nodes = F.normalize(
                teacher_nodes.flatten(2).transpose(1, 2).float(), dim=2
            )
            similarity = torch.bmm(
                teacher_nodes, teacher_nodes.transpose(1, 2)
            )

        centres = F.normalize(cluster_centres.detach().float(), dim=1)
        normalized_online = F.normalize(online_nodes, dim=2)
        logits = torch.einsum("bnc,kc->bnk", normalized_online, centres)
        probabilities = (logits / self.prediction_temperature).softmax(dim=2)

        node_count = grid * grid
        neighbor_count = min(self.neighbors, node_count - 1)
        diagonal = torch.eye(
            node_count, dtype=torch.bool, device=online_code.device
        )
        positive_similarity, positive_indices = similarity.masked_fill(
            diagonal.unsqueeze(0), -2.0
        ).topk(neighbor_count, dim=2)
        expanded_probabilities = probabilities[:, :, None, :].expand(
            -1, -1, neighbor_count, -1
        )
        positive_probabilities = torch.gather(
            probabilities[:, None, :, :].expand(-1, node_count, -1, -1),
            2,
            positive_indices[:, :, :, None].expand(
                -1, -1, -1, probabilities.shape[-1]
            ),
        )
        midpoint = 0.5 * (expanded_probabilities + positive_probabilities)
        attraction = 0.5 * (
            (
                expanded_probabilities
                * (
                    expanded_probabilities.clamp_min(1e-6).log()
                    - midpoint.clamp_min(1e-6).log()
                )
            ).sum(dim=3)
            + (
                positive_probabilities
                * (
                    positive_probabilities.clamp_min(1e-6).log()
                    - midpoint.clamp_min(1e-6).log()
                )
            ).sum(dim=3)
        )
        # A merely "least bad" negative neighbour must not become attractive.
        positive_weights = positive_similarity.clamp(0.0, 1.0)
        attraction_loss = (
            attraction * positive_weights
        ).sum() / positive_weights.sum().clamp_min(1e-6)

        source, target = self._grid_edges(grid, online_code.device)
        with torch.no_grad():
            boundary_similarity = (
                teacher_nodes[:, source] * teacher_nodes[:, target]
            ).sum(dim=2)
            negative_threshold = torch.quantile(
                boundary_similarity,
                self.negative_quantile,
                dim=1,
                keepdim=True,
            )
            negative_mask = boundary_similarity <= negative_threshold
            negative_weights = (
                (1.0 - boundary_similarity) * 0.5
            ).clamp(0.0, 1.0)
            negative_weights = negative_weights * negative_mask.float()
        overlap = (
            probabilities[:, source] * probabilities[:, target]
        ).sum(dim=2)
        repulsion = F.relu(overlap - self.negative_margin)
        repulsion_loss = (
            repulsion * negative_weights
        ).sum() / negative_weights.sum().clamp_min(1e-6)

        total = (
            self.attraction_weight * attraction_loss
            + self.repulsion_weight * repulsion_loss
        )
        statistics = {
            "attraction": float(attraction_loss.detach().cpu()),
            "repulsion": float(repulsion_loss.detach().cpu()),
            "weighted_attraction": float(
                (self.attraction_weight * attraction_loss).detach().cpu()
            ),
            "weighted_repulsion": float(
                (self.repulsion_weight * repulsion_loss).detach().cpu()
            ),
            "positive_edges_per_node": float(neighbor_count),
            "negative_edge_fraction": float(
                negative_mask.float().mean().cpu()
            ),
        }
        return total, statistics


__all__ = ["SparseRegionGraphConsistency"]
