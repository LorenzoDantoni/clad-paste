from __future__ import annotations

from typing import Optional, Union, Any, Mapping, List

from feature_extractor import (
    CustomFeatureExtractor,
)
from PIL import Image

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import get_model
from torchvision.models.feature_extraction import create_feature_extractor


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
                model_name, last_layer, torch.device("cpu"), frozen=self.is_teacher
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


class Stfpm(nn.Module):

    BACKBONE_HYPERPARAMS = [
        "weights_name",
        "backbone_model_name",
        "student_bootstrap_layer",
        "ad_layers",
    ]

    HYPERPARAMS = [
        *BACKBONE_HYPERPARAMS,
        "input_size",
        "output_size",
        "epochs",
        "category",
        "seed",
    ]

    def __init__(
        self,
        backbone_model_name: Optional[str] = None,
        input_size=(224, 224),
        output_size=(224, 224),
        ad_layers: Optional[Union[List[int], List[str]]] = None,
        weights="IMAGENET1K_V2",
        student_bootstrap_layer: Optional[int] = None,
    ):
        """
        This class manages the STFPM AD model
        Either provide a load_path to load a checkpoint or provide the backbone_model_name
        and layers to create a new model.

        Parameters:
            backbone_model_name: name of the model to be used as backbone such as "resnet18" or "mobilenet_v2"
            input_size: tuple with the input size of the images
            output_size: tuple with the output size of the model output
            ad_layers: list of integers representing the layers to be used for anomaly detection
            weights: None if the model is not pretrained, otherwise "DEFAULT" or "IMAGENET1K_V2" etc.
            student_bootstrap_layer: index of the layer to be used as bootstrap for the student model.
                The teacher computes the feature maps up and including this layer, and the output of this
                layer is used as input for the student model. If False, the student model is trained from
                the input image.
        """
        super(Stfpm, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        # backbone params
        self.weights_name = weights
        self.backbone_model_name = backbone_model_name
        if student_bootstrap_layer is False:
            student_bootstrap_layer = None
        self.student_bootstrap_layer = student_bootstrap_layer
        self.ad_layers = self.__layers_to_idxs__(ad_layers, backbone_model_name)
        
        # training params
        self.seed: Optional[int] = None
        self.epochs: Optional[int] = None
        self.category: Optional[str] = None

        if all(
            [
                self.backbone_model_name,
                self.ad_layers,
            ]
        ):
            self.__define_backbones__()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # add all the hyperparameters to the state dict
        for p in self.HYPERPARAMS:
            state_dict[p] = getattr(self, p)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # load the hyperparameters
        for p in self.HYPERPARAMS:
            setattr(self, p, state_dict[p])
        # load the backbone models
        self.__define_backbones__()
        return super().load_state_dict(state_dict, strict=strict)

    @staticmethod
    def __layers_to_idxs__(layers, model=None):
        """
        Defaults to converting the layer strings to integers, but can be extended to
        convert the layer names to indexes for specific models.
        """
        # here we can add model names to the list, and convert the layers to indexes
        # in an appropriate way for each model that needs it
        # if model in []:
        #     return
        if layers is None:
            return None
        return [int(l) for l in layers]

    def __define_backbones__(self):
        assert (
            self.ad_layers is not None
        ), "The layers to use for anomaly detection must be defined."
        # the teacher is a pretrained model, use the default best weights
        self.teacher = StfpmBackbone(
            self.backbone_model_name,
            self.ad_layers,
            weights=self.weights_name,
            bootstrap_idx=self.student_bootstrap_layer,
            is_teacher=True,
        )

        # the student's weights are initialized randomly
        self.student = StfpmBackbone(
            self.backbone_model_name,
            self.ad_layers,
            weights=None,
            bootstrap_idx=self.student_bootstrap_layer,  # shared layers
            is_teacher=False,
        )

    def model_filename(self):
        assert (
            self.ad_layers is not None
        ), "The layers to use for anomaly detection must be defined."
        layers = "_".join(map(str, self.ad_layers))
        boot = (
            f"_boots{self.student_bootstrap_layer}"
            if self.student_bootstrap_layer
            else ""
        )
        return f"{self.backbone_model_name}_{self.epochs}ep_{self.weights_name}_{layers}{boot}_s{self.seed}.pth.tar"

    def forward(self, batch_imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass

        Parameters: 
        ----------
            - batch_imgs: input images tensors
        
        Returns:
        --------
            - tuple: [0] teacher features, [1] student features if the model is in training mode
            - anomaly maps if the model is in evaluation mode
        """

        # the teacher is freezed
        self.teacher.eval()
        with torch.no_grad():
            t_feat, bootstrap_feat = self.teacher(batch_imgs)
        
        # perform PaSTe or not
        x = batch_imgs if self.student_bootstrap_layer is None else bootstrap_feat
        s_feat, _ = self.student(x)

        if self.training:
            return t_feat, s_feat
        else:
            return self.post_process(t_feat, s_feat)

    def post_process(self, t_feat, s_feat) -> torch.Tensor:
        
        """
        This method actually produces the anomaly maps for evaluation purposes

        Parameters: 
        ----------
            - t_feat: teacher features maps
            - s_feat: student features maps

        Returns: 
        --------
            - anomaly maps

        """

        device = t_feat[0].device
        score_maps = torch.tensor([1.0], device=device)
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(
                sm, size=self.output_size, mode="bilinear", align_corners=False
            )
            # aggregate score map by element-wise product
            score_maps = score_maps * sm

        anomaly_scores = torch.max(score_maps.view(score_maps.size(0), -1), dim=1)[0]
        return score_maps, anomaly_scores

    def eval(self, *args, **kwargs):
        self.teacher.eval()
        self.student.eval()
        return super().eval(*args, **kwargs)


