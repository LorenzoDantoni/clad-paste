from __future__ import annotations
from typing import Optional, Union, Any, Mapping, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.stfpm_paste_add.stfpm_backbone import StfpmBackbone

from src.models.stfpm_add.timm import FeatureExtractor
from src.models.stfpm_add.anomaly_map import AnomalyMapGenerator


def create_stfpm_paste(strategy, img_shape, parameters):
    device_id = strategy.parameters.get("device_id")
    backbone_model_name = strategy.parameters.get("backbone_model_name")
    ad_layers = strategy.parameters.get("ad_layers")
    weights = strategy.parameters.get("weights")
    student_bootstrap_layer = parameters.get("student_bootstrap_layer")
    # img_size = strategy.parameters.get("img_size")

    device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")

    st = StfpmPaste(
        device, strategy, backbone_model_name, img_shape[1:], img_shape[1:], ad_layers, weights, student_bootstrap_layer
    )
    st = st.to(device)

    return st, device


class StfpmPaste(nn.Module):

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
            device: torch.device,
            strategy,
            backbone_model_name: Optional[str] = None,
            input_size: tuple[int, int] = (224, 224),
            output_size: tuple[int, int] = (224, 224),
            ad_layers: Optional[Union[List[int], List[str]]] = None,
            weights: str ="IMAGENET1K_V2",
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
                layer is used as input for the student model.
                If False, the student model is trained from the input image.
        """
        super(StfpmPaste, self).__init__()

        self.device = device
        self.strategy = strategy

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

        self.anomaly_map_generator = AnomalyMapGenerator(strategy, image_size=self.output_size)

    def forward(self, batch_imgs: torch.Tensor, sample_strategy: str, index_training: int, test, caller, batch_task_labels) -> tuple[dict[str, Tensor], dict[str, Tensor]] | Tensor:
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

        # the teacher is frozen
        self.teacher.eval()
        self.sliced_teacher.eval()
        with torch.no_grad():
            if sample_strategy == "compressed_replay_paste" and index_training != 0 and not test:
                t_feat, _ = self.sliced_teacher(batch_imgs)
            else:
                t_feat, bootstrap_feat = self.teacher(batch_imgs)

        # perform PaSTe or not
        if sample_strategy == "compressed_replay_paste" and index_training != 0 and not test:
            s_feat, _ = self.student(batch_imgs)
        else:
            x = batch_imgs if self.student_bootstrap_layer is None else bootstrap_feat
            s_feat, _ = self.student(x)

        t_feat = self.convert_features_to_dict(t_feat)
        s_feat = self.convert_features_to_dict(s_feat)

        if self.training:
            return t_feat, s_feat
        else:
            return self.anomaly_map_generator(
                teacher_features=t_feat, student_features=s_feat, caller=caller, batch_task_labels=batch_task_labels)
            # return self.post_process(t_feat, s_feat)


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
        self.sliced_teacher.eval()
        return super().eval(*args, **kwargs)


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
            self.device,
            self.backbone_model_name,
            self.ad_layers,
            weights=self.weights_name,
            bootstrap_idx=self.student_bootstrap_layer,
            is_teacher=True,
            sliced_teacher=False,
        )

        # the student's weights are initialized randomly
        self.student = StfpmBackbone(
            self.device,
            self.backbone_model_name,
            self.ad_layers,
            weights=None,
            bootstrap_idx=self.student_bootstrap_layer,  # shared layers
            is_teacher=False,
            sliced_teacher=False,
        )

        self.sliced_teacher = StfpmBackbone(
            self.device,
            self.backbone_model_name,
            self.ad_layers,
            weights=self.weights_name,
            bootstrap_idx=self.student_bootstrap_layer,
            is_teacher=True,
            sliced_teacher=True,
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


    def convert_features_to_dict(self, feature_maps: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Convert a list of feature maps into a dictionary with the layer names as keys.

        Args:
            feature_maps (list[torch.Tensor]): A list of feature maps.

        Returns:
            dict[str, torch.Tensor]: A dictionary where the keys are the layer names as strings
                                      and the values are the corresponding feature maps.
        """
        if len(self.ad_layers) != len(feature_maps):
            raise ValueError("ad_layers and feature_maps must have the same length.")

        feature_dict = {f"layer{layer}": feature_map for layer, feature_map in zip(self.ad_layers, feature_maps)}

        return feature_dict