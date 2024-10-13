"""
This class represent a feature extractor built on top of a custom backbone. The backbone name is passed as input to the constructor. 
Based on the model name and the layers indexes it will consider the correct layers for the feature extraction. 
"""

from __future__ import annotations

import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

# Add the 'project' directory to the Python path

from extractors.mcunet.mcunet.model_zoo import net_id_list, build_model

#from backbones.micronet.micronet import micronetBB
#from micromind import PhiNet

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
    "phinet_0.9_0.5_4_downsampling"
)

TORCH_BACKBONES = (
    "vgg19_bn",
    "resnet18",
    "wide_resnet50_2",
    "efficientnet_b5",
    "mobilenet_v2"
)


class CustomFeatureExtractor:

    def __init__(self, model_name: str, layers_idx: list, device: torch.device, frozen: bool = True):
        """
        Constructor

        Args:
            model_name (str): name of the backbone to use
            layers_idx (list): list of layers identifiers
            device (torch.device): device to be used
        """

        self.model_name = model_name
        self.layers_idx = layers_idx
        self.device = device

        #¢heck for backbone support
        if model_name not in OTHERS_BACKBONES + TORCH_BACKBONES:
            raise Exception(f"The backbone: {model_name} is not supported for feature extraction")

        #check for mcunet backbone
        if "mcunet" in model_name:
            if model_name in net_id_list:
                self.model, _, _ = build_model(net_id=model_name, pretrained=True)

                #attach the hook
            self.attach_hook()

        # check for micronet backbone
        # if "micronet" in model_name:
        #     self.model = micronetBB(device, torch.load(f"backbones/micronet/weights/{self.model_name}.pth"))
        #
        #     #attach the hook
        #     self.attach_hook()
        #
        # # check for phinet backbone
        # if "phinet" in model_name:
        #     self.load_phinet()
        #
        #     #attach the hook
        #     self.attach_hook()

        if model_name in TORCH_BACKBONES:
            self.model = CustomFeatureExtractor.get_feature_extractor(model_name, layers_idx)

        #load the model to the device
        self.model = self.model.to(self.device)

        if frozen:
            #freeze the model
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    # def load_phinet(self):
    #
    #     """
    #     This function handles the phinet backbone loading
    #     """
    #
    #     if self.model_name == 'phinet_2.3_0.75_5':
    #         self.model = PhiNet(input_shape=(3, 224, 224), alpha=2.3, beta=0.75, t_zero=5, include_top=True,
    #                             num_classes=1000, divisor=8)
    #         self.model.load_state_dict(torch.load('backbones/phinet/new_phinet_small_71.pth.tar')["state_dict"])
    #     elif self.model_name == 'phinet_1.2_0.5_6_downsampling':
    #         self.model = PhiNet(input_shape=(3, 224, 224), num_layers=7, alpha=1.2, beta=0.5, t_zero=6,
    #                             downsampling_layers=[4, 5, 7], include_top=True, num_classes=1000, divisor=8)
    #         self.model.load_state_dict(
    #             torch.load('backbones/phinet/new_phinet_divisor8_v2_downsampl.pth.tar')["state_dict"])
    #     elif self.model_name == 'phinet_0.8_0.75_8_downsampling':
    #         self.model = PhiNet(input_shape=(3, 224, 224), num_layers=7, alpha=0.8, beta=0.75, t_zero=8,
    #                             downsampling_layers=[4, 5, 7], include_top=True, num_classes=1000, divisor=8)
    #         self.model.load_state_dict(torch.load('backbones/phinet/new_phinet_divisor8_v3.pth.tar')["state_dict"])
    #     elif self.model_name == 'phinet_1.3_0.5_7_downsampling':
    #         self.model = PhiNet(input_shape=(3, 224, 224), num_layers=7, alpha=1.3, beta=0.5, t_zero=7,
    #                             downsampling_layers=[4, 5, 7], include_top=True, num_classes=1000, divisor=8)
    #         self.model.load_state_dict(torch.load('backbones/phinet/phinet_13057DS.pth.tar')["state_dict"])
    #     elif self.model_name == 'phinet_0.9_0.5_4_downsampling_deep':
    #         self.model = PhiNet(input_shape=(3, 224, 224), num_layers=9, alpha=0.9, beta=0.5, t_zero=4,
    #                             downsampling_layers=[4, 5, 7], include_top=True, num_classes=1000, divisor=8)
    #         self.model.load_state_dict(torch.load('backbones/phinet/phinet_09054DSDE.pth.tar')["state_dict"])
    #     elif self.model_name == 'phinet_0.9_0.5_4_downsampling':
    #         self.model = PhiNet(input_shape=(3, 224, 224), num_layers=7, alpha=0.9, beta=0.5, t_zero=4,
    #                             downsampling_layers=[4, 5, 7], include_top=True, num_classes=1000, divisor=8)
    #         self.model.load_state_dict(torch.load('backbones/phinet/phinet_09054DS.pth.tar')["state_dict"])

    @staticmethod
    def get_feature_extractor(backbone: str, return_nodes):
        """Get the feature extractor from the backbone CNN.

        Args:
            backbone (str): Backbone CNN network
            return_nodes (list[str]): A list of return nodes for the given backbone.

        Raises:
            NotImplementedError: When the backbone is efficientnet_b5
            ValueError: When the backbone is not supported

        Returns:
            GraphModule: Feature extractor.
        """
        model = getattr(torchvision.models, backbone)(pretrained=True)
        feature_extractor = create_feature_extractor(model=model, return_nodes=return_nodes)

        return feature_extractor

    def attach_hook(self, bootstrap_idx=0):

        def hook(module, input, output):
            self.features.append(output)

        feature_layers = None

        if "mcunet" in self.model_name:
            feature_layers = list(self.model.blocks)

        if "micronet" in self.model_name:
            feature_layers = list(self.model.features)

        if "phinet" in self.model_name:
            feature_layers = list(self.model._layers)

        for idx in self.layers_idx:
            assert idx - bootstrap_idx > 0, "Invalid layer index"
            feature_layers[idx - bootstrap_idx].register_forward_hook(hook)

    def __call__(self, batch: torch.Tensor) -> list[torch.Tensor]:

        """
        batch:Tensor [1,3,224,224]
        Returns:
        -------
        lista di feature maps, una per ogni layer specificato nel costruttore
        """

        if self.model_name in TORCH_BACKBONES:
            return list(self.model(batch).values())

        else:
            self.features = []
            self.model(batch.to(self.device))
            return self.features


# def test_feature_extractor():
#     # Choose the backbone and layers to extract features from (e.g., layers of resnet18)
#     model_name = 'mcunet-in3'
#     layers_idx = [2, 6, 14]
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     dummy_input = torch.randn(1, 3, 224, 224)
#     dummy_input = dummy_input.to(device)
#
#     # freeze the backbone by default
#     feature_extractor = CustomFeatureExtractor(model_name, layers_idx, device)
#
#     # Run the dummy input through the extractor and get feature maps
#     feature_maps = feature_extractor(dummy_input)
#
#     # Print the shape of each feature map to verify the output
#     for i, feature_map in enumerate(feature_maps):
#         print(f"Feature map {i} from layer {layers_idx[i]} has shape: {feature_map.shape}")
#
#
# test_feature_extractor()