import os

import numpy as np
import torch
import random
import itertools
import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
                    'importance': top_importance,
                    'all_features_importance': importance_normalized,
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
    optimal_n_components_tasks = {}

    if strategy.parameters.get("sample_strategy") == "multi_task":
        num_tasks = 10

    # Iterate through each task to extract top features
    for index_training in range(num_tasks):
        # Get the label of the current task
        task_label = labels_map[index_training]
        optimal_n_components_tasks[task_label] = {}

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

        optimal_n_components_tasks[task_label] = optimal_n_components

        # Step 3: Apply PCA per patch and compute the top important features
        print(f"\nComputing PCA per patch ({task_label}) ...")
        patch_top_features = apply_pca_per_patch(
            teacher_embeddings_dict,
            optimal_n_components,
            percentage_top_features_to_retain  # Retain a percentage of the top features per patch
        )

        all_tasks_top_results[task_label] = patch_top_features

        # plot_feature_importance_across_layers(patch_top_features, task_label)

    return all_tasks_top_results, all_task_teacher_embeddings, optimal_n_components_tasks


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

            mask = torch.zeros((C, H, W), device=device)
            # mask = torch.ones((C, H, W), device=device)

            # Iterate through each patch (i, j) to apply top feature filtering
            for i in range(H):
                for j in range(W):
                    top_features = patch_results[(i, j)]['indices']
                    mask[top_features, i, j] = 1.0  # Mark important features as 1 in the mask
                    # mask[top_features, i, j] = 0.0

            feature_masks[task_label][layer] = mask

    return feature_masks


def combine_indices_across_tasks(all_tasks_top_results):
    """
    Gathers the combined top feature indices for each patch and layer across all tasks.

    Args:
        all_tasks_top_results (dict): A dictionary where keys are task labels and values are dictionaries
                                      containing layer-wise top patch features for each task.

    Returns:
        combined_indices (dict): A dictionary where keys are (layer, patch) tuples and values are the combined
                                 sets of indices across tasks.
    """
    combined_indices = {}

    # Iterate over task labels and gather the combined indices for each [layer, (i, j)]
    for task_label, layers_results in all_tasks_top_results.items():
        for layer, patch_results in layers_results.items():
            if layer not in combined_indices:
                combined_indices[layer] = {}

            for (i, j), top_features in patch_results.items():
                if (i, j) not in combined_indices[layer]:
                    combined_indices[layer][(i, j)] = set()

                # Add the top feature indices of the current task for the current patch
                combined_indices[layer][(i, j)].update(top_features['indices'])

    return combined_indices


def select_and_combine_filtered_features(all_tasks_teacher_embeddings, combined_indices):
    """
    Selects and concatenates the filtered features across all tasks based on the combined indices.
    The output is an array ready to be fed into t-SNE.

    Args:
        all_tasks_teacher_embeddings (dict): Dictionary containing embeddings for all tasks.
                                             Keys are task labels, and values are dictionaries with
                                             layer-wise embeddings.
                                             Shape for each layer is (B, C, H, W).
        combined_indices (dict): Dictionary where keys are (layer, patch) and values are the combined indices
                                 across tasks.

    Returns:
        all_filtered_features (ndarray): The concatenated filtered features across all tasks
                                          Shape (total_batches, total_filtered_features).
        task_labels_array (ndarray): Array of task labels corresponding to each sample for classification or clustering.
    """
    all_filtered_features = []
    task_labels_array = []

    # Iterate over each task and select features based on the combined indices
    for task_label, task_embeddings in all_tasks_teacher_embeddings.items():
        task_filtered_features = []

        # Iterate over each layer and patch to select the corresponding features
        for layer, patch_indices in combined_indices.items():
            embeddings = task_embeddings[layer]  # Shape (B, C, H, W)
            B, C, H, W = embeddings.shape

            for (i, j), indices in patch_indices.items():
                # Select the features for the current patch (i, j) using the combined indices
                selected_features = embeddings[:, list(indices), i, j]  # Shape (B, len(indices))
                task_filtered_features.append(selected_features)

        # Concatenate the filtered features for the current task along the feature dimension
        task_filtered_features = np.concatenate(task_filtered_features, axis=1)  # Shape (B, total_selected_features)

        # Add to the list of all filtered features
        all_filtered_features.append(task_filtered_features)

        # Create a label array for this task, repeating the task label for each batch sample
        task_labels_array.extend([task_label] * B)

    # Stack the features for all tasks together
    all_filtered_features = np.vstack(all_filtered_features)  # Shape (total_batches, total_filtered_features)

    # Convert task_labels_array to a numpy array for easier manipulation later
    task_labels_array = np.array(task_labels_array)

    return all_filtered_features, task_labels_array


def plot_tsne(all_filtered_features, task_labels_array, percentage_top_features_to_retain, seed):
    """
        Computes and plots t-SNE for the given features and task labels.

        Args:
            all_filtered_features (ndarray): The concatenated filtered features across all tasks.
                                             Shape (total_samples, total_filtered_features).
            task_labels_array (ndarray): Array of task labels corresponding to each sample.
            percentage_top_features_to_retain (int): Number of top percentage of features retained.
            seed (int): The random seed for t-SNE.

        Returns:
            tsne_results (ndarray): The 2D t-SNE embedding for the input features.
    """

    # Step 1: Perform t-SNE on the filtered features
    tsne = TSNE(n_components=2, random_state=seed)
    tsne_results = tsne.fit_transform(all_filtered_features)  # Shape (total_samples, 2)

    # Step 2: Create a scatter plot of the t-SNE results, colored by task labels
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1], hue=task_labels_array, palette="deep", legend="full", s=60
    )

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.title(f't-SNE plot of top {percentage_top_features_to_retain}% features - Patch-wise PCA ')

    layer_dir = os.path.join('plots', 'stfpm_tsne')
    os.makedirs(layer_dir, exist_ok=True)
    plot_filename = os.path.join(layer_dir, f'tsne_patch_wise_pca_top{percentage_top_features_to_retain}%.png')
    plt.savefig(plot_filename)
    plt.close()

    return tsne_results

def plot_optimal_components_per_layer_vertical(optimal_n_components_tasks):
    """
    Plots a grouped bar plot for each layer, showing the optimal number of PCA components per task.

    Args:
        optimal_n_components_tasks (dict): Dictionary where keys are task labels and values are dictionaries of
                                           optimal PCA components per layer.
                                           Example: {task_label: {layer_name: n_components, ...}, ...}
    """

    tasks = list(optimal_n_components_tasks.keys())
    layers = list(next(iter(optimal_n_components_tasks.values())).keys())

    # Set up a grid of subplots, one for each layer
    num_layers = len(layers)
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 18), sharex=True)

    for i, layer in enumerate(layers):
        # Gather the component counts for the current layer across tasks
        n_components = [optimal_n_components_tasks[task][layer] for task in tasks]

        # Plot the grouped bar plot for the current layer
        axes[i].bar(tasks, n_components, color=plt.cm.tab10.colors[:len(tasks)])
        axes[i].set_title(f"{layer}", fontsize=14, fontweight='bold')
        axes[i].set_ylabel("Optimal PCA Components", fontsize=12)
        axes[i].tick_params(axis='x', rotation=45, labelsize=14)
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout(h_pad=2)

    layer_dir = os.path.join('plots', 'pca_components')
    os.makedirs(layer_dir, exist_ok=True)
    plot_filename = os.path.join(layer_dir, f'pca_components_retention_by_layer.png')
    plt.savefig(plot_filename)
    plt.close()


def plot_optimal_components_per_layer_horizontal(optimal_n_components_tasks):
    """
    Plots a grouped bar plot for each layer in a 1-row, 3-column layout,
    showing the optimal number of PCA components per task.

    Args:
        optimal_n_components_tasks (dict): Dictionary where keys are task labels and values are dictionaries of
                                           optimal PCA components per layer.
                                           Example: {task_label: {layer_name: n_components, ...}, ...}
    """

    tasks = list(optimal_n_components_tasks.keys())
    layers = list(next(iter(optimal_n_components_tasks.values())).keys())

    # Set up a grid of subplots with 1 row and 3 columns
    num_layers = len(layers)
    fig, axes = plt.subplots(1, num_layers, figsize=(20, 6), sharey=True)

    for i, layer in enumerate(layers):
        n_components = [optimal_n_components_tasks[task][layer] for task in tasks]

        # Plot the grouped bar plot for the current layer
        axes[i].bar(tasks, n_components, color=plt.cm.tab10.colors[:len(tasks)])
        axes[i].set_title(f"{layer}", fontsize=14, fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45, labelsize=14)
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.6)

    # Set shared y-axis label for all subplots
    fig.text(0, 0.5, 'Optimal PCA Components', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout(w_pad=2)  # Increase spacing between plots

    layer_dir = os.path.join('plots', 'stfpm_pca_components')
    os.makedirs(layer_dir, exist_ok=True)
    plot_filename = os.path.join(layer_dir, f'pca_components_retention_by_layer.png')
    plt.savefig(plot_filename)
    plt.close()

def plot_feature_usage_distribution(all_tasks_top_results):
    # num_tasks = 10
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.ravel()  # Flatten axes for easier indexing

    for i, (task_label, top_features) in enumerate(all_tasks_top_results.items()):
        feature_counts = {}

        for layer_features in top_features.values():
            for patch_data in layer_features.values():
                indices = patch_data['indices']
                for index in indices:
                    feature_counts[index] = feature_counts.get(index, 0) + 1

        sorted_indices = sorted(feature_counts.keys())
        sorted_counts = [feature_counts[idx] for idx in sorted_indices]
        axes[i].bar(sorted_indices, sorted_counts)
        axes[i].set_title(f'{task_label}', fontdict={'fontsize': 20})
        axes[i].set_xlabel('Feature Index', fontdict={'fontsize': 16})
        if i % 5 == 0:
            axes[i].set_ylabel('Frequency', fontdict={'fontsize': 16})

    plt.suptitle("Feature Usage Distribution per Task")
    plt.tight_layout()

    layer_dir = os.path.join('plots', 'stfpm_feature_usage_distribution')
    os.makedirs(layer_dir, exist_ok=True)
    plot_filename = os.path.join(layer_dir, f'feature_usage_distribution_per_task.png')
    plt.savefig(plot_filename)
    plt.close()


def plot_layerwise_feature_usage_heatmap(all_tasks_top_results):
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.ravel()  # Flatten axes for easier indexing

    for i, (task_label, top_features) in enumerate(all_tasks_top_results.items()):
        # Prepare layer-feature count matrix for the heatmap
        layer_feature_counts = {}

        for layer_name, layer_features in top_features.items():
            feature_counts = {}
            for patch_data in layer_features.values():
                indices = patch_data['indices']
                for index in indices:
                    feature_counts[index] = feature_counts.get(index, 0) + 1

            layer_feature_counts[layer_name] = feature_counts

        heatmap_data = pd.DataFrame(layer_feature_counts).T
        heatmap_data_sorted = heatmap_data[sorted(heatmap_data.columns)]
        sns.heatmap(heatmap_data_sorted, ax=axes[i], cmap="YlGnBu", cbar=True, cbar_kws={'shrink': 0.6})
        axes[i].set_title(f'{task_label}')
        axes[i].set_xlabel('Feature Index')

    plt.suptitle("Top Feature Usage Heatmap Across Layers and Tasks")
    plt.tight_layout()

    layer_dir = os.path.join('plots', 'stfpm_feature_usage_distribution')
    os.makedirs(layer_dir, exist_ok=True)
    plot_filename = os.path.join(layer_dir, f'top_feature_usage_heatmap.png')
    plt.savefig(plot_filename)
    plt.close()


def plot_feature_importance_across_layers(patch_top_features, task_label):
    # Organize importance values by layer
    layer_importance_data = {layer: [] for layer in patch_top_features.keys()}

    for layer, patches in patch_top_features.items():
        for patch_info in patches.values():
            # Append all importance values from this patch to the layer's list
            layer_importance_data[layer].extend(patch_info['importance'])

    # Prepare data for the box plot
    layers = list(layer_importance_data.keys())
    importance_values = [layer_importance_data[layer] for layer in layers]

    # Create box plot
    plt.figure(figsize=(12, 6))
    plt.boxplot(importance_values, labels=layers, patch_artist=True, notch=True)
    # plt.xlabel('Layers')
    plt.ylabel('Normalized Feature Importance')
    plt.title(f'Distribution of Normalized Feature Importance Across Layers of {task_label}')
    plt.xticks(rotation=45)
    plt.tight_layout()

    layer_dir = os.path.join('plots', 'distribution_norm_feature_importance')
    os.makedirs(layer_dir, exist_ok=True)
    plot_filename = os.path.join(layer_dir, f'distribution_norm_feature_importance_layers_{task_label}.png')
    plt.savefig(plot_filename)
    plt.close()


def plot_feature_importance_across_tasks(all_tasks_top_results):
    fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=True)
    fig.suptitle('Distribution of Normalized Feature Importance Across Layers for Each Task', fontsize=16)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    for task_idx, (task_label, patch_top_features) in enumerate(all_tasks_top_results.items()):
        # Organize importance values by layer for the current task
        layer_importance_data = {layer: [] for layer in patch_top_features.keys()}

        for layer, patches in patch_top_features.items():
            for patch_info in patches.values():
                layer_importance_data[layer].extend(patch_info['importance'])

        layers = list(layer_importance_data.keys())
        importance_values = [layer_importance_data[layer] for layer in layers]

        # Create box plot in the corresponding subplot
        ax = axes[task_idx]
        ax.boxplot(importance_values, labels=layers, patch_artist=True, notch=True)
        ax.set_title(f'{task_label}', fontdict={'fontsize': 20})
        ax.tick_params(axis='x', labelsize=16)

        # Set y-axis label only for the leftmost subplots for readability
        if task_idx % 5 == 0:
            ax.set_ylabel('Normalized Feature Importance', fontdict={'fontsize': 16})

    # Adjust layout to make space for titles and labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    layer_dir = os.path.join('plots', 'stfpm_distribution_norm_feature_importance')
    os.makedirs(layer_dir, exist_ok=True)
    plot_filename = os.path.join(layer_dir, f'all_tasks_distribution_norm_feature_importance_layers.png')
    plt.savefig(plot_filename)
    plt.close()


def plot_multiple_layers_heatmap(all_tasks_top_results, use_top_features_only=True):
    num_layers = 3
    num_cols = 3
    num_rows = (num_layers + num_cols - 1) // num_cols

    for task_label, patch_top_features in all_tasks_top_results.items():
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        fig.suptitle(f"{task_label}", fontsize=22)
        axes = axes.flatten()

        for idx, (layer, patches) in enumerate(patch_top_features.items()):
            ax = axes[idx]

            H, W = max(i for i, j in patches.keys()) + 1, max(j for i, j in patches.keys()) + 1
            importance_matrix = np.zeros((H, W))

            for (i, j), data in patches.items():
                if use_top_features_only:
                    # Sum or average the importance of the top features
                    importance_matrix[i, j] = np.sum(data['importance'])
                else:
                    # Sum of all features' importance (if you have this data)
                    importance_matrix[i, j] = np.sum(data['all_features_importance'])

            # Plot the heatmap for the current layer
            sns.heatmap(importance_matrix, annot=False, cmap="cividis", ax=ax, cbar_kws={'label': 'Total Importance'})
            ax.set_title(f"{layer}", fontdict={'fontsize': 20})

            ax.set_xticks([0, W - 1])
            ax.set_yticks([0, H - 1])
            ax.set_xticklabels([0, W - 1])
            ax.set_yticklabels([0, H - 1])

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=0)

            # ax.set_xlabel("Patch Width")
            # ax.set_ylabel("Patch Height")

        # Remove any empty axes (if num_layers isn't a perfect multiple of num_cols)
        for i in range(num_layers, len(axes)):
            fig.delaxes(axes[i])

        # Adjust layout for better spacing
        plt.tight_layout()
        layer_dir = os.path.join('plots', 'test_cividis_top_feature_importance_heatmap_resnet')
        os.makedirs(layer_dir, exist_ok=True)
        plot_filename = os.path.join(layer_dir, f'{task_label}_feature_importance_heatmap.png')
        plt.savefig(plot_filename)

        plt.close()

def no_range_find_and_remove_feature_conflicts_patch_wise(all_tasks_top_results):
    """
    Finds feature conflicts across all tasks based on shared feature usage in a patch-wise manner,
    and removes the conflicting features directly from the task results.

    Args:
        all_tasks_top_results (dict): A dictionary where keys are task labels and values are dictionaries containing
                                      layer-wise top patch features for each task.

    Returns:
        filtered_tasks_top_results (dict): A dictionary where conflicting features have been removed.
    """

    feature_usage = {}

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

    # Step 3: Remove conflicting features without range overlap check
    for feature_key, tasks in feature_conflicts.items():
        layer, (i, j), feature_index = feature_key

        # Update the feature list of each task involved in the conflict
        for task_label in tasks:
            indices = all_tasks_top_results[task_label][layer][(i, j)]['indices']
            new_indices = indices[indices != feature_index]
            all_tasks_top_results[task_label][layer][(i, j)]['indices'] = new_indices

    # Return the filtered top features after removing conflicts
    return all_tasks_top_results
