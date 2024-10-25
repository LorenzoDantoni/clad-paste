import numpy as np
import torch
import random
import itertools

from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import Tensor


def extract_embeddings_from_teacher(strategy, dataloader):
    """
    Extract embeddings for all images in the dataloader using the trained student model.

    Args:
        strategy: The strategy object that contains configurations
        dataloader (DataLoader): The dataloader containing images from the current task.

    Returns:
        embeddings_dict (dict): A dictionary where keys are layer names and values are numpy arrays of concatenated embeddings.
    """

    # Initialize a dictionary to hold embeddings for each specified layer
    if strategy.parameters['architecture'] == "stfpm_paste":
        ad_indexes = strategy.trainer.ad_model.teacher.ad_layers_idxs
        teacher_embeddings_dict = {f"layer{idx}": [] for idx in ad_indexes}
    else:
        teacher_embeddings_dict = {layer: [] for layer in strategy.trainer.ad_model.teacher.layers}

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch[0].to(strategy.trainer.ad_model.device)

            if strategy.parameters['architecture'] == "stfpm_paste":
                t_feat, _ = strategy.trainer.ad_model.teacher(images)
                teacher_features: dict[str, Tensor] = {f"layer{idx}": feature_map for idx, feature_map in zip(ad_indexes, t_feat)}
            else:
                teacher_features: dict[str, Tensor] = strategy.trainer.ad_model.teacher(images)

            # Store the features (embeddings) from each layer
            for layer, features in teacher_features.items():
                teacher_embeddings_dict[layer].append(features.detach().cpu().numpy())

    # Each batch produces a batch of feature maps. Need to combine them into a single array.
    # final shape == (total_number_of_images, channels, height, width)
    for layer in teacher_embeddings_dict:
        teacher_embeddings_dict[layer] = np.concatenate(teacher_embeddings_dict[layer], axis=0)

    return teacher_embeddings_dict


def apply_pca_per_patch(embeddings_dict, optimal_n_components, percentage_top_features_to_retain=1):
    """
    Applies PCA to each patch (i, j) for every layer using the optimal number of components.

    Args:
        embeddings_dict (dict): Dictionary with layer-wise embeddings.
                                Shape for each layer is (B, C, H, W).
        optimal_n_components (dict): Dictionary where keys are layer names and values are the optimal
                                     number of components determined for each layer.
        percentage_top_features_to_retain (float): Percentage of top features to retain (e.g., 1 for top 1%).

    Returns:
        patch_top_features (dict): A dictionary containing the top important features for each patch
                                   in every layer. Keys are layer names, values are dictionaries with
                                   patch positions (i, j) as keys and important feature indices as values.
    """
    patch_top_features = {}
    # normalizer = Normalizer()
    # scaler = StandardScaler()

    for layer, embeddings in embeddings_dict.items():
        B, C, H, W = embeddings.shape
        patch_top_features[layer] = {}

        n_components = optimal_n_components[layer]
        print(f"\nApplying PCA on {layer} using {n_components} components per patch")

        for i in range(H):
            for j in range(W):
                # shape (B, C)
                data = embeddings[:, :, i, j]

                # Normalize the patch data
                # data = normalizer.fit_transform(data)
                # data = scaler.fit_transform(data)

                # Apply PCA on the patch data using the optimal number of components
                pca = PCA(n_components=n_components).fit(data)

                # Compute feature importance as a weighted sum of PCA components based on variance ratio
                components = pca.components_
                explained_variance_ratio = pca.explained_variance_ratio_
                importance = np.abs(components.T).dot(explained_variance_ratio)

                # Normalize the importance to sum to 1
                importance_normalized = importance / importance.sum()

                # Calculate the threshold for top features based on the percentage to retain
                threshold = np.percentile(importance_normalized, 100 - percentage_top_features_to_retain)
                top_indices = np.where(importance_normalized >= threshold)[0]
                top_importance = importance_normalized[top_indices]

                # Store the top indices and importance for patch (i, j)
                patch_top_features[layer][(i, j)] = {
                    'indices': top_indices,
                    'importance': top_importance
                }

    return patch_top_features


def approximate_optimal_number_pca_components(strategy, embeddings_dict, variance_threshold=0.80, percentage_random_patches=0.25):
    """
    Approximates the optimal number of components for applying PCA on each patch for every layer.

    Args:
        strategy: The strategy object that contains configurations
        embeddings_dict (dict): Dictionary with layer-wise embeddings.
                                Shape for each layer is (B, C, H, W).
        variance_threshold (float): Variance threshold to determine n_components.
        percentage_random_patches: Percentage of randomly chosen patches

      Returns:
        optimal_n_components (dict): A dictionary where keys are layer names and values are the
                                     99th percentile of the number of components required across all patches.
    """
    pca_results = {layer: [] for layer in embeddings_dict.keys()}
    # normalizer = Normalizer()
    # scaler = StandardScaler()

    for layer, embeddings in embeddings_dict.items():
        B, C, H, W = embeddings.shape
        patches_indices = [(i, j) for i in range(H) for j in range(W)]

        # Take 1/4 of the total patches
        num_random_patches = int(H * W * percentage_random_patches)

        print(f"\n\n{layer}: using {num_random_patches}/{H * W} ({percentage_random_patches * 100}%) patches to compute the approximated optimal number of PCA components")

        random.seed(strategy.seed)
        random_patches = random.sample(patches_indices, num_random_patches)

        for (i, j) in random_patches:
            # shape (B, C)
            data = embeddings[:, :, i, j]

            # Normalize the patch data
            # data = normalizer.fit_transform(data)
            # data = scaler.fit_transform(data)

            # Apply PCA on the patch data
            pca = PCA().fit(data)

            # Compute cumulative variance and determine number of components for variance_threshold% variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

            pca_results[layer].append(n_components)

    # Calculate 99th percentile for each layer's components
    optimal_n_components = {}
    for layer, components_list in pca_results.items():
        components = int(np.percentile(components_list, 99))
        optimal_n_components[layer] = components
        print(f"\n{layer} approximated optimal number of components to retain {variance_threshold * 100}% variance: {components}")

    return optimal_n_components


def extract_top_features_from_each_task(strategy, num_tasks, train_stream, percentage_top_features_to_retain, labels_map):
    """
    Extracts the top important features for each task by applying PCA per patch on teacher embeddings and
    retaining the specified percentage of important features.
    The result is stored in a dictionary that maps each task to its corresponding layer-wise top patch features.

    Args:
        strategy (object): The strategy object that contains configurations like batch size, trainer, and parameters.
        num_tasks (int): The total number of tasks to iterate over.
        train_stream (list): A list of training datasets for each task.
        percentage_top_features_to_retain (float): The percentage of top features to retain after PCA for each patch.
        labels_map (dict): A dictionary mapping task indices to task labels.

    Returns:
        all_tasks_top_results (dict): A dictionary where keys are task labels and values are dictionaries
                                      containing layer-wise top patch features for each task.
        all_task_teacher_embeddings (dict): A dictionary where keys are task labels and values are dictionaries
                                            containing layer-wise numpy arrays of concatenated embeddings extracted
                                            from the teacher model.
        """

    # Initialize a dictionary to store the top features for all tasks
    all_tasks_top_results = {}
    all_task_teacher_embeddings = {}

    if strategy.parameters.get("sample_strategy") == "multi_task":
        num_tasks = 10

    # Iterate through each task to extract top features
    for index_training in range(num_tasks):
        # Get the label of the current task
        task_label = labels_map[index_training]

        print(f"\nPreprocessing task {task_label}")

        # Load the current task's training dataset
        if strategy.parameters.get("sample_strategy") == "multi_task":
            current_train_dataset = train_stream[0].datasets[index_training]
        else:
            current_train_dataset = train_stream[index_training]
        current_train_data_loader = DataLoader(
            current_train_dataset,
            batch_size=strategy.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=strategy.parameters["architecture"] != "storig"
        )

        # Step 1: Extract teacher embeddings for the current task
        print(f"\nExtracting embeddings from teacher ({task_label}) ...")
        teacher_embeddings_dict = extract_embeddings_from_teacher(strategy, current_train_data_loader)
        all_task_teacher_embeddings[task_label] = teacher_embeddings_dict

        # Step 2: Approximate the optimal number of PCA components for each patch
        print(f"\nComputing optimal number of components for PCA ({task_label}) ...")
        optimal_n_components = approximate_optimal_number_pca_components(
            strategy,
            teacher_embeddings_dict,
            variance_threshold=0.80,  # Retain 80% of the variance in the data
            percentage_random_patches=0.10  # Sample 10% of patches for PCA component estimation
        )

        # Step 3: Apply PCA per patch and compute the top important features
        print(f"\nComputing PCA per patch ({task_label}) ...")
        patch_top_features = apply_pca_per_patch(
            teacher_embeddings_dict,
            optimal_n_components,
            percentage_top_features_to_retain  # Retain a percentage of the top features per patch
        )

        all_tasks_top_results[task_label] = patch_top_features

    return all_tasks_top_results, all_task_teacher_embeddings


def find_and_remove_feature_conflicts_range_patch_wise(all_tasks_top_results, all_task_teacher_embeddings):
    """
    Finds detailed feature conflicts across all tasks based on shared feature usage in a patch-wise manner,
    and removes the conflicting features after another filter based on the range of the features
    directly from the task results.

    Args:
        all_task_teacher_embeddings: A dictionary where keys are task labels
                                    and values are dictionaries containing layer-wise numpy arrays of concatenated
                                    embeddings extracted from the teacher model.
        all_tasks_top_results (dict): A dictionary where keys are task labels and values are dictionaries containing
                                      layer-wise top patch features for each task.

    Returns:
        filtered_tasks_top_results (dict): A dictionary where conflicting features have been removed.
    """

    feature_usage = {}
    fake_conflicts = 0  # Counter for "fake" conflicts resolved by range overlap
    # total_conflicts = 0

    # Step 1: Track feature usage across tasks
    for task_label, layers_results in all_tasks_top_results.items():
        for layer, patch_results in layers_results.items():
            for (i, j), top_features in patch_results.items():
                feature_indices = top_features['indices']

                # Track feature usage by (layer, patch, feature) tuple
                for feature_index in feature_indices:
                    feature_key = (layer, (i, j), feature_index)
                    if feature_key not in feature_usage:
                        feature_usage[feature_key] = set()
                    feature_usage[feature_key].add(task_label)

    # Step 2: Identify conflicts (features used by more than one task in the same patch and layer)
    feature_conflicts = {feature_key: tasks for feature_key, tasks in feature_usage.items() if len(tasks) > 1}

    # Step 3: Apply range-based conflict filtering for features used across tasks
    def is_conflict_based_on_range(feature_key, tasks):
        """
        Checks if a feature is really in conflict by calculating the overlap of ranges across tasks.
        If the overlap is greater than 30%, the feature is not considered a conflict.

        Args:
            feature_key (tuple): The (layer, patch, feature) tuple.
            tasks (set): Set of task labels where the feature is used.

        Returns:
            bool: True if the feature is in conflict, False otherwise.
        """

        layer, (i, j), feature_index = feature_key
        ranges = []

        # Step 3.1: Calculate ranges (min, max) for the feature across tasks
        for task in tasks:
            # Get the feature values
            task_embeddings = all_task_teacher_embeddings[task][layer][:, feature_index, i, j]

            avg = task_embeddings.mean().item()
            std = task_embeddings.std().item()

            min_range = avg - 3 * std
            max_range = avg + 3 * std

            ranges.append((min_range, max_range))

        # Step 3.2: Compare the ranges across all task pairs
        for r1, r2 in itertools.combinations(ranges, 2):
            min_overlap = max(r1[0], r2[0])  # Max of the lower bounds
            max_overlap = min(r1[1], r2[1])  # Min of the upper bounds
            overlap_size = max(0, max_overlap - min_overlap)
            total_size = max(r1[1], r2[1]) - min(r1[0], r2[0])
            overlap_percentage = (overlap_size / total_size) if total_size > 0 else 0

            # If there's more than 30% overlap, it's not considered a conflict
            if overlap_percentage > 0.3:
                return False  # Not a conflict

        return True  # It's a conflict if less than 30% overlap across all tasks

    # Step 4: Remove conflicting features based on range filter (loop only on conflicts)
    for feature_key, tasks in feature_conflicts.items():
        layer, (i, j), feature_index = feature_key

        # Check if the conflict is "resolved" by range overlap
        if not is_conflict_based_on_range(feature_key, tasks):
            fake_conflicts += 1
        else:
            # Update the feature list of each task involved in the conflict
            for task_label in tasks:
                indices = all_tasks_top_results[task_label][layer][(i, j)]['indices']
                new_indices = indices[indices != feature_index]
                all_tasks_top_results[task_label][layer][(i, j)]['indices'] = new_indices

    num_conflicts = len(feature_conflicts.keys())
    total_features = len(feature_usage)
    true_conflicts = num_conflicts - fake_conflicts

    print(f"\nTotal features: {total_features}")
    print(f"Total features initially marked as conflicts: {num_conflicts}")
    print(f"'Fake' conflicts (resolved by range analysis): {fake_conflicts}")
    print(f"True conflicts: {true_conflicts}")

    conflict_percentage = int((true_conflicts / total_features) * 100)
    print(f"Final number of feature conflicts: {true_conflicts}/{total_features} ---- {conflict_percentage}%\n")

    # for (layer, patch, feature), tasks in feature_conflicts.items():
    #     print(f"Layer {layer}, Patch {patch}, Feature {feature} is used by tasks: {', '.join(tasks)}")

    # Return the filtered top features after removing conflicts
    return all_tasks_top_results


def create_feature_mask_for_each_task(filtered_tasks_top_results, all_task_teacher_embeddings, device):
    """
    Creates a binary mask for each task, layer, and patch, where the important (non-conflicting) features
    are marked with 1, and the rest are marked with 0.

    Args:
       filtered_tasks_top_results (dict): The filtered top feature results for each task after removing conflicts.
       all_task_teacher_embeddings (dict): A dictionary containing teacher embeddings for each task and layer, used
                                           to get the dimensions of each layer for mask creation.
       device: pytorch device

    Returns:
       feature_masks (dict): A dictionary where keys are task labels, and values are dictionaries containing
                             layer-wise binary masks for each task.
   """
    feature_masks = {}

    for task_label, layers_results in filtered_tasks_top_results.items():
        feature_masks[task_label] = {}

        # Iterate through each layer for the current task
        for layer, patch_results in layers_results.items():
            teacher_embeddings_layer = all_task_teacher_embeddings[task_label][layer]

            if isinstance(teacher_embeddings_layer, np.ndarray):
                teacher_embeddings_layer = torch.from_numpy(teacher_embeddings_layer).to(device)
            else:
                teacher_embeddings_layer = teacher_embeddings_layer.to(device)

            B, C, H, W = teacher_embeddings_layer.shape

            # mask = torch.zeros((C, H, W), device=device)
            mask = torch.ones((C, H, W), device=device)

            # Iterate through each patch (i, j) to apply top feature filtering
            for i in range(H):
                for j in range(W):
                    top_features = patch_results[(i, j)]['indices']
                    # mask[top_features, i, j] = 1.0  # Mark important features as 1 in the mask
                    mask[top_features, i, j] = 0.0  # Mark important features as 1 in the mask

            feature_masks[task_label][layer] = mask

    return feature_masks