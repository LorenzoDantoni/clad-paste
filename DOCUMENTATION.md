# Documentation for Compressed Replay on the PaSTe Framework in CLAD

## Command

```bash
python main.py --parameters_path train_stfpm_paste.json --seed 43
```

## Configuration Directory

### New Configuration Files
- **`stfpm_paste.json`**: Defines the model configuration for the PaSTe framework.
- **`train_stfpm_paste.json`**: Defines the training configuration for the PaSTe framework.

#### Added Parameters
- **`"sample_strategy"`**: `"compressed_replay_paste"` (strategy for handling replay)
- **`"backbone_model_name"`**: `"mcunet-in3"` (name of the backbone model)
- **`"weights"`**: `"IMAGENET1K_V2"` (pre-trained weights)
- **`"student_bootstrap_layer"`**: `5` (index of the backbone layer used for feature extraction)

## Models Directory

### Added Files

- **`stfpm_paste.py`**:
  - Implements the PaSTe framework (`StfpmPaste` class).

- **`stfpm_backbone.py`**:
  - Implements the backbone model for the PaSTe framework (`StfpmBackbone` class).

- **`feature_extractor.py`**:
  - Custom feature extractor used by the `StfpmBackbone` class.

## Trainer Directory

### Added File

- **`trainer_stfpm_paste.py`**:
  - Implements the training and evaluation logic for the PaSTe framework.

## Utilities Directory
### Added File

- **`utility_feature_importance.py`**:
  - Contains utility functions related to feature importance:
    - **Feature Importance Functions**:
      - Extract embeddings from the teacher.
      - Select top features for each task.
      - Apply PCA per patch.
      - Remove conflicting features.
      - Create masks to select only the most important features.
    - **t-SNE Computation**:
      - Combine indices of important features across tasks.
      - Combine all features extracted by the model.
      - Generate t-SNE plots.
    - **EDA Analysis**:
      - Visualize various insightful plots.

## Dataset Directory

### Modified File: `dataset.py`

- **Class `MemoryDataset`**:
  - Updated the `__getitem__` method to incorporate compressed replay logic.
  - Retrieves images/features stored in replay memory in pickle format.
  - Decompresses features if they were previously compressed.

## Memory Directory

### Modified File: `memory.py`

- **New Class `MemoryCompressedReplayPaste`**:
  - Handles the new compressed replay logic.
  - Saves feature maps for each task seen so far.

- **Updated Method**:
  - `create_batch_data` (during training): Enables the compressed replay strategy.

## Main Scripts

### `main_pca.py`

- Applies the feature importance strategy during multi-task continual learning.
- Selects the top 10% of features for each patch.
- Can be used during both inference and training.

### `main_eda.py`

- Does not perform training.
- Focuses on EDA, generating insightful plots.
- Uncomment specific methods based on requirements.

### `main.py`

- Implements default compressed replay without feature importance.
- Supports the following compression parameters:
  - `"PCA"`: PCA-based feature compression.
  - `"scale_quantization"`: 8-bit feature quantization.
  - `""`: No compression.
