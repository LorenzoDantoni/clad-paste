from typing import Optional, List

import torch
import torch.nn as nn
from torchvision.models import get_model

from src.models.stfpm_paste_add.feature_extractor import CustomFeatureExtractor

OTHERS_BACKBONES = (
    "mcunet-in3",
    "micronet-m0",
    "micronet-m1",
    "micronet-m2",
    "micronet-m3",
    "phinet_2.3_0.75_5",
    "phinet_1.2_0.5_6_downsampling",
    "phinet_0.8_0.75_8_downsampling",
    "phinet_1.3_0.5_7_downsampling",
    "phinet_0.9_0.5_4_downsampling_deep",
    "phinet_0.9_0.5_4_downsampling",
)

TORCH_BACKBONES = (
    "vgg19_bn",
    "resnet18",
    "wide_resnet50_2",
    "efficientnet_b5",
    "mobilenet_v2",
)


class StfpmBackbone(nn.Module):
    def __init__(
        self,
        device: torch.device,
        model_name: str,
        ad_layers_idxs: List[int],
        weights: Optional[str],
        bootstrap_idx: int = None,
        is_teacher: bool = False,
    ):
        """
        This class manages the STFPM backbones of teacher and student.

        Parameters:
        -----------
            - model_name: name of the model to be used for teacher and student
            - ad_layers_idxs: list of integers representing the layers to be used for anomaly detection
            - weights: None if the model is not pretrained, otherwise "DEFAULT" or "IMAGENET1K_V2" etc.
            - bootstrap_idx: index of the boostrap layer
            - is_teacher: boolean if $this model is a student or a teacher (the structure will be different)
        """
        super().__init__()

        if bootstrap_idx is not None:
            if bootstrap_idx > min(ad_layers_idxs):
                raise ValueError(
                    "The bootstrap layer must be before the first AD layer.",
                    f"Bootstrap layer: {bootstrap_idx}, AD layers: {ad_layers_idxs}",
                )
        if bootstrap_idx is False:
            bootstrap_idx = None

        self.bootstrap_idx = bootstrap_idx
        self.ad_layers_idxs = ad_layers_idxs
        self.is_teacher = is_teacher
        self.model_name = model_name

        # get a list of layers that compose the model
        if model_name in TORCH_BACKBONES:
            model = get_model(model_name, weights=weights)

            if model_name == "mobilenet_v2":
                feat_extraction_layers = list(model.children())[0]

            if model_name == "wide_resnet50_2":
                feat_extraction_layers = list(model.children())
                feat_extraction_layers = [
                    feat_extraction_layers[0],  # layer0
                    nn.Sequential(*feat_extraction_layers[1:5]),  # layer1
                    feat_extraction_layers[5],  # layer2
                    feat_extraction_layers[6],  # layer3
                    feat_extraction_layers[7],  # layer4
                ]
        else:
            backbones_last_layer = {
                "phinet_1.2_0.5_6_downsampling": [9],  # 0 to 9
                "mcunet-in3": [17],  # 0 to 17
                "micronet-m1": [7],  # 0 to 7
            }
            last_layer = backbones_last_layer[model_name]
            feature_extractor = CustomFeatureExtractor(
                model_name, last_layer, device, frozen=self.is_teacher
            )

            feat_extraction_layers = list(feature_extractor.model.children())

            if "mcunet" in model_name:
                feat_extraction_layers = [
                    torch.nn.Sequential(*feat_extraction_layers[:2])
                ] + feat_extraction_layers[2:]

        if is_teacher:
            # use all the layers until the last desired layer
            layers_slice = slice(max(ad_layers_idxs) + 1)
            self.layer_offset = 0
        else:
            # use the layers between the one next to the bootstrap layer and the last desired layer
            if bootstrap_idx is not None and bootstrap_idx is not False:
                bootstrap_idx += 1
            else:
                bootstrap_idx = None
            layers_slice = slice(bootstrap_idx, max(ad_layers_idxs) + 1)
            self.layer_offset = (
                0 if self.bootstrap_idx is None else 1 + self.bootstrap_idx
            )

        #for debug
        print(f"is teacher: {is_teacher}, layers slice: {layers_slice}")
        self.model = torch.nn.Sequential(*feat_extraction_layers[layers_slice])

    def forward(self, x: torch.Tensor) -> tuple[List[torch.Tensor], torch.Tensor]:

        """
        Forward method

        Parameters:
        -----------
            - x: input tensor

        Returns:
        -------
            - tuple:
                [0] : List of torch Tensor with the list of extracted features
                [1] : torch Tensor with the extracted features from the boostrap layer
        """

        res = []
        bootstrap_feat = None
        # Forward the input through each layer of the model
        for i, (_, module) in enumerate(
            self.model._modules.items(), start=self.layer_offset
        ):
            x = module(x)
            # Save the output of the desired layers
            if i in self.ad_layers_idxs:
                res.append(x)
            if self.bootstrap_idx is not None and (i == self.bootstrap_idx):
                bootstrap_feat = x.clone()

        return res, bootstrap_feat
