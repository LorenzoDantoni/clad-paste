# CLAD: Continual Learning Anomaly Detection 

The entire project has been setup to be easily run in an Anaconda environment. Before running the code, the environment needs to be set by using the <code>adcl.yml</code> file present inside the main directory of the repository. In order to set it up enter the following line in Anaconda Prompt: 

<code> conda env create -f adcl.yml </code>

After that you need to download and place the training datasets in the correct folders. Proceed with the following instructions step-by-step:

1. Put MVTec dataset classes in <code>./data/mvtec folder</code> so as to finally have bottle, cable, capsule folders etc.
2. Download Describable Textures Dataset (DTD) from the following link: https://www.robots.ox.ac.uk/~vgg/data/dtd/ and put it in folder <code>./anomaly_dataset</code> for the purpose of running DRAEM. As a result you will have the folders: images, imdb, labels.
3. Download Tiny ImageNet from the following link: http://cs231n.stanford.edu/tiny-imagenet-200.zip and put it in folder <code>./tiny-imagenet-200</code> for the purpose of running EfficientAD and the STFPM. As a result there will be the following folders: test, train, val and files: wnids, words.
4. In <code>./configurations/credentials.json</code> insert the "project_name" under which the logs on Wandb account will be created and your Wandb account "api-token".

After that you can start to train the models by setting the CL strategy that you prefer:

1. In the <code>./configurations/</code> folder you can find the files named <code>test_[model_name]_ideal_replay</code> where you can set the CL strategy (replay, naive, multi_task, single_model) you want to run (in the "sample_strategy" field) and replay memory buffer size (in the "mem_size" field).

NB: the models STFPM, CFA, DRAEM, FastFlow and EfficientAD are not memory-bank based and thus in order to train them with a replay strategy you have to set in their configurations json files the field "sample_strategy" to "replay" and the field "mem_size" to 300 or 100 or to the quantity that you want. 
Patchcore and Padim are memory-bank based so the continual learning strategy adopted is not replay. In order to train them in a CL strategy you have to set the flag "cl" to True in their .json file under the directory <code>./configurations/models/</code> and in their configuration file you have to specify the "cl" strategy in the "sample_strategy" field. For PatchCore you can also set the "mem_size_cl" field in the json file in the <code>./configurations/models/</code> directory, which defines the size of the memory bank. 

2. Execute the training with this command: 

<code>python main.py --parameters_path path_to_the_conf_json_file --seed seed</code>

and by specifyng the correct configuration json file and the seed for the training.

The results will be recorded on Wandb under the indicated project_name, while image results will be stored in the folder output that is automatically created when code is run
