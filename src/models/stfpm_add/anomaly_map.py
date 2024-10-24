"""Anomaly Map Generator for the STFPM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor, nn


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(self, strategy, image_size: ListConfig | tuple) -> None:
        super().__init__()

        self.strategy = strategy

        # TODO: where it is used???
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)

        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)

    def compute_layer_map(self, teacher_features: Tensor, student_features: Tensor, layer: str, caller: str, batch_task_labels: torch.Tensor) -> Tensor:
        """Compute the layer map.

        Args:
          teacher_features (Tensor): Teacher features
          student_features (Tensor): Student features
          layer (str): Layer name for feature extraction.
          caller (str): Indicates the calling method ("test_epoch" or "evaluate_data").
          batch_task_labels (torch.Tensor): Task labels for each sample in the batch.

        Returns:
          Tensor: The anomaly score map.
        """
        def apply_mask_and_normalize_per_sample(teacher_feats: Tensor, student_feats: Tensor, task_label: str) -> tuple:
            """Apply mask to teacher and student features and normalize them."""
            if hasattr(self, 'feature_masks') and task_label in self.feature_masks:
                mask = self.feature_masks[task_label][layer]

                # Ensure mask matches the shape of (B, C, H, W)
                # mask_expanded = mask.unsqueeze(0).expand_as(teacher_feats)

                # Apply the mask to filter both teacher and student features
                filtered_teacher_feats = teacher_feats * mask
                filtered_student_feats = student_feats * mask

                # Normalize filtered features
                norm_teacher_feats = F.normalize(filtered_teacher_feats, dim=1)
                norm_student_feats = F.normalize(filtered_student_feats, dim=1)

                return norm_teacher_feats, norm_student_feats

            norm_teacher_features = F.normalize(teacher_feats, dim=1)
            norm_student_features = F.normalize(student_feats, dim=1)

            return norm_teacher_features, norm_student_features

        norm_teacher_features = []
        norm_student_features = []

        # Iterate over each sample in the batch
        for i in range(batch_task_labels.size(0)):
            task_idx = batch_task_labels[i].item()
            task_label = self.strategy.labels_map[task_idx]

            norm_teacher_feat, norm_student_feat = apply_mask_and_normalize_per_sample(teacher_features[i], student_features[i], task_label)

            norm_teacher_features.append(norm_teacher_feat)
            norm_student_features.append(norm_student_feat)

        # Stack the features back into a tensor
        norm_teacher_features = torch.stack(norm_teacher_features)
        norm_student_features = torch.stack(norm_student_features)

        layer_map = 0.5 * torch.norm(norm_teacher_features - norm_student_features, p=2, dim=-3, keepdim=True) ** 2
        layer_map = F.interpolate(layer_map, size=self.image_size, align_corners=False, mode="bilinear")

        return layer_map

    # def compute_layer_map(self, teacher_features: Tensor, student_features: Tensor) -> Tensor:
    #     """Compute the layer map based on cosine similarity.
    #
    #     Args:
    #       teacher_features (Tensor): Teacher features
    #       student_features (Tensor): Student features
    #
    #     Returns:
    #       Anomaly score based on cosine similarity.
    #     """
    #     # default p=2 and dim=1
    #     norm_teacher_features = F.normalize(teacher_features)
    #     norm_student_features = F.normalize(student_features)
    #
    #     layer_map = 0.5 * torch.norm(norm_teacher_features - norm_student_features, p=2, dim=-3, keepdim=True) ** 2
    #     layer_map = F.interpolate(layer_map, size=self.image_size, align_corners=False, mode="bilinear")
    #     return layer_map

    def compute_anomaly_map(
        self, teacher_features: dict[str, Tensor], student_features: dict[str, Tensor], caller: str, batch_task_labels
    ) -> torch.Tensor:
        """Compute the overall anomaly map via element-wise production the interpolated anomaly maps.

        Args:
          teacher_features (dict[str, Tensor]): Teacher features
          student_features (dict[str, Tensor]): Student features

        Returns:
          Final anomaly map
        """
        batch_size = list(teacher_features.values())[0].shape[0]
        anomaly_map = torch.ones(batch_size, 1, self.image_size[0], self.image_size[1])
        for layer in teacher_features.keys():
            layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer], layer, caller, batch_task_labels)
            anomaly_map = anomaly_map.to(layer_map.device)
            anomaly_map *= layer_map

        return anomaly_map

    def forward(self, **kwargs: dict[str, Tensor]) -> torch.Tensor:
        """Returns anomaly map.

        Expects `teach_features` and `student_features` keywords to be passed explicitly.

        Example:
            >>> anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(hparams.model.input_size))
            >>> output = self.anomaly_map_generator(
                    teacher_features=teacher_features,
                    student_features=student_features
                )

        Raises:
            ValueError: `teach_features` and `student_features` keys are not found

        Returns:
            torch.Tensor: anomaly map
        """

        if not ("teacher_features" in kwargs and "student_features" in kwargs):
            raise ValueError(f"Expected keys `teacher_features` and `student_features. Found {kwargs.keys()}")

        teacher_features: dict[str, Tensor] = kwargs["teacher_features"]
        student_features: dict[str, Tensor] = kwargs["student_features"]

        # added for Feature Importance
        caller = kwargs["caller"]
        batch_task_labels = kwargs["batch_task_labels"]

        return self.compute_anomaly_map(teacher_features, student_features, caller, batch_task_labels)
