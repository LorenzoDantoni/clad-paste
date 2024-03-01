The entire project has been setup to be easily run in Anaconda env - Spyder. Before running the code environment needs to be set whose .yml file is provided in adcl_paper folder. In order to set it up enter the following line in Anaconda Prompt: 

conda env create -f adcl.yml

Open Spyder(adcl) and proceed with the following instructions step-by-step:

1. Put MVTec dataset classes in ./data/mvtec folder so as to finally have bottle, cable, capsule folders etc.
2. Download Describable Textures Dataset (DTD) from the following link: https://www.robots.ox.ac.uk/~vgg/data/dtd/ and put it in folder anomaly_dataset for the purpose of running DRAEM. As a result you will have the folders: images, imdb, labels.
3. Download Tiny ImageNet from the following link: http://cs231n.stanford.edu/tiny-imagenet-200.zip and put it in folder tiny-imagenet-200 for the purpose of running EfficientAD and ST. As a result there will be the following folders: test, train, val and files: wnids, words.
4. In configurations/credentials.json insert the "project_name" under which the logs on Wandb account will be created and your Wandb account "api-token".
5. In configurations folder you can find the files named test_[model_name]_ideal_replay where you can set the CL strategy (replay, naive, multi_task, single_model) you want to run ("sample_strategy") and rep
lay memory buffer size ("mem_size").
6. In the function init_execute written in src/utilities/utility_main.py in code line section 276-291 put Wandb account entity.
7. In main.py in line 30 put the correct .json file of the model to be run.
8. Run main.py
9. The results will be recorded on Wandb under the indicated project_name, while image results will be stored in the folder output that is automatically created when code is run
