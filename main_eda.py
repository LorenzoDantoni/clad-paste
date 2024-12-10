import argparse
import os
import sys
import shutil
from datetime import datetime
import warnings
# ADDED
import time
import os

ROOT = ".."
sys.path.append(ROOT)
sys.path.append(ROOT + "/pytorch_pix2pix")

# from src.models.fastflow import *
from src.models.cfa import *
from src.strategy_ad import *
# from src.trainer.trainer_fastflow import *
from src.trainer.trainer_cfa import *
from src.datasets import *
from src.utilities.utility_main import *
from src.utilities import utility_logging
from src.utilities.utility_models import *
from torch.utils.data import DataLoader
from src.utilities.utility_feature_importance import extract_top_features_from_each_task, \
    find_and_remove_feature_conflicts_range_patch_wise, create_feature_mask_for_each_task, plot_tsne, \
    combine_indices_across_tasks, select_and_combine_filtered_features, plot_optimal_components_per_layer_vertical, \
    plot_optimal_components_per_layer_horizontal, plot_feature_usage_distribution, plot_layerwise_feature_usage_heatmap, \
    plot_feature_importance_across_tasks, plot_multiple_layers_heatmap, \
    no_range_find_and_remove_feature_conflicts_patch_wise

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description="Parser to take filepaths")
parser.add_argument("--parameters_path", type=str, nargs="?", action='store', help="parameters path",
                    default="test_cfa_ideal_replay.json")  # loads specific parameters from .json file for specific model

# ONLY FOR DEBUG
# parser.add_argument("--credentials_path", type=str, nargs="?", action = 'store', help="credentials path", default="credentials.json")                #load the credentials for wandb logging
# parser.add_argument("--default_path", type=str, nargs="?", action = 'store', help="default parameters path", default="common_param.json")            #common parameters for the training

parser.add_argument("--seed", type=int, nargs="?", action='store', help="seed",
                    default=random.randint(1, 10000))  # set seed

# load paths to the parameters variable (model_specific, credentials for Neptune, common params)
args = parser.parse_args()
path = 'configurations'
parameters_path = os.path.join(path, args.parameters_path).replace('\\', '/')
credentials_path = os.path.join(path, "credentials.json").replace('\\', '/')
default_path = os.path.join(path, "common_param.json").replace('\\', '/')
seed = args.seed

print(f"seed: {seed}")
print(f"parametes_path: {parameters_path}")
print(f"credentials_path: {credentials_path}")
print(f"default_path: {default_path}")

# seed = 43

# Get wandb run object,parameters and available device
run, parameters, device = init_execute(credentials_path, default_path, parameters_path, seed)
project_name = run.project
experiment_name = run.id

now = datetime.now()  # current date and time
date_time = now.strftime("%d_%m_%Y__%H-%M-%S")
path_logs = os.path.join(f"logs/{project_name}/{experiment_name}_{date_time}").replace('\\',
                                                                                       '/')  # Neptune path for logging data
print(f"path_logs: {path_logs}")
utility_logging.create_paths([path_logs])

filename = os.path.basename(parameters_path)
dst = os.path.join(path_logs, filename).replace('\\',
                                                '/')  # /logs/{project_name}/{experiment_name}_{date_time}/test_fast_flow_standard.json
shutil.copyfile(parameters_path, dst)  # copy parameters (specific for the model) to Neptune

# Load Dataset
channels, dataset_name, num_tasks, task_order = parameters["channels"], parameters["dataset_name"], parameters[
    "num_tasks"], parameters["task_order"]
complete_train_dataset, complete_test_dataset, train_stream, test_stream = load_and_split_dataset(parameters,
                                                                                                  dataset_name,
                                                                                                  num_tasks,
                                                                                                  task_order)  # returns train and test dataset in default task order, and then in specified order for CL
# output values: MVTecDataset (it contains list of x,y...), MVTecDataset(same), list(of Subsets), list(of Subsets)

labels_map = create_new_labels_map(labels_datasets[dataset_name], task_order,
                                   num_tasks)  # put strings of classes' names in desired task order
print(f"labels_map: {labels_map}")

# Create Strategy
if isinstance(complete_train_dataset[0][0], dict):
    input_size = complete_train_dataset[0][0]["image"].shape  # input_size=256
else:
    # torch.Size([3, 256, 256])
    input_size = complete_train_dataset[0][0].shape
print(f"input_size: {input_size}")

original_stdout = sys.stdout  # Save a reference to the original standard output
filepath = os.path.join(path_logs, 'model_info.txt').replace('\\',
                                                             '/')  # cretes model_info.txt file within the created project on Neptune

with open(filepath, 'w') as f:
    sys.stdout = f  # change output default destination
    strategy = create_strategy(parameters, run, labels_map, device, path_logs,
                               input_size)  # creates strategy.trainer,.test_loss_function, .input_size, .device
    ''' self.parameters = parameters

        self.num_tasks = num_tasks
        self.task_order = task_order
        self.num_epochs = num_epochs
        self.labels_map = labels_map
        self.path_logs = path_logs
        self.run = run'''
sys.stdout = original_stdout

strategy.seed = seed

import copy

original_complete_train_dataset, original_complete_test_dataset, original_train_stream, original_test_stream = copy.deepcopy(complete_train_dataset), copy.deepcopy(complete_test_dataset), copy.deepcopy(train_stream), copy.deepcopy(test_stream)
complete_train_dataset, complete_test_dataset, train_stream, test_stream = manage_dataset(strategy, parameters,
                                                                                          complete_train_dataset,
                                                                                          complete_test_dataset,
                                                                                          train_stream, test_stream)
num_tasks = strategy.num_tasks
elapsed_time = 0
init_strategy_variables(strategy, complete_train_dataset, complete_test_dataset, train_stream, test_stream,
                        original_complete_train_dataset, original_complete_test_dataset, original_train_stream,
                        original_test_stream, labels_map, run, path_logs, elapsed_time)

sample_strategy = strategy.parameters.get("sample_strategy")
test_only_seen_tasks = strategy.parameters.get("test_only_seen_tasks")

if sample_strategy == "multi_task" and test_only_seen_tasks:
    raise ValueError("test_only_seen_tasks is True but you are in multi_task mode")

ad_layers = strategy.parameters.get("ad_layers")
ad_layers_list = [f"layer{layer_idx}" for layer_idx in ad_layers]

percentage_top_features_to_retain = 1

# Step 1
all_tasks_top_results, all_task_teacher_embeddings, optimal_n_components_tasks = extract_top_features_from_each_task(
    strategy, num_tasks, test_stream, percentage_top_features_to_retain, labels_map)

# EDA plots: uncomment the methods based on what plot you want
##################################################################
# plot_feature_importance_across_tasks(all_tasks_top_results)
# plot_layerwise_feature_usage_heatmap(all_tasks_top_results)
# plot_feature_usage_distribution(all_tasks_top_results)
# plot_optimal_components_per_layer_horizontal(optimal_n_components_tasks)

# combined_indices = combine_indices_across_tasks(all_tasks_top_results)
# all_filtered_features, task_labels_array = select_and_combine_filtered_features(all_task_teacher_embeddings, combined_indices)
# plot_tsne(all_filtered_features, task_labels_array, percentage_top_features_to_retain, seed)

# Remove feature conflicts (no range filter)
# all_tasks_top_results = find_and_remove_feature_conflicts_range_patch_wise(
#     all_tasks_top_results, all_task_teacher_embeddings
# )

# Remove feature conflicts (added range filter)
# all_tasks_top_results = no_range_find_and_remove_feature_conflicts_patch_wise(all_tasks_top_results)

# plot_multiple_layers_heatmap(all_tasks_top_results)
##################################################################

run.log({"Training Time": strategy.elapsed_time})
print(f"Training time: {strategy.elapsed_time} seconds")
# run["Finished"].log(True)
run.finish()
