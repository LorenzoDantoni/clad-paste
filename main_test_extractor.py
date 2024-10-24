import torch

from src.models.stfpm_paste import StfpmPaste
from src.models.stfpm_paste_add.feature_extractor import CustomFeatureExtractor


def test_feature_extractor():
    # Choose the backbone and layers to extract features from (e.g., layers of resnet18)
    model_name = 'mcunet-in3'
    layers_idx = [2, 6, 14]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_input = dummy_input.to(device)

    # freeze the backbone by default
    feature_extractor = CustomFeatureExtractor(model_name, layers_idx, device)

    # Run the dummy input through the extractor and get feature maps
    feature_maps = feature_extractor(dummy_input)

    print(feature_extractor.model)

    # Print the shape of each feature map to verify the output
    for i, feature_map in enumerate(feature_maps):
        print(f"Feature map {i} from layer {layers_idx[i]} has shape: {feature_map.shape}")


def test_paste():
    model_name = 'mcunet-in3'
    layers_idx = [6, 10, 14]
    boostrap_idx = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_input = dummy_input.to(device)

    st = StfpmPaste(
        device, model_name, ad_layers=layers_idx, student_bootstrap_layer=boostrap_idx
    ).to(device)

    st.train()

    tf, sf = st(dummy_input)
    return



test_paste()