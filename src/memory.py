import torch
import os
from PIL import Image
import numpy as np
import pickle
from numpy.random import default_rng
from sklearn.decomposition import PCA

from src.datasets import MemoryDataset
from src.utilities.utility_logging import from_sample_to_dict,save_pickle,save_img,load_pickle, save_img_draem
from src.utilities.utility_pix2pix import create_summary, create_summary_by_numpy, forward_test, produce_scaled_A_from_B, forward_test
from src.utilities.utility_ad import *
from src.utilities.utility_models import produce_output_given_model_and_sample, produce_output_given_model_from_noise
from src.utilities.utility_quantization import quantize_feature_map


def create_memory(strategy, task_order, dir_experiment_path, mem_size, current_task, type_strategy):
    '''
    function that instantiate the class to apply the specified sample strategy

    Args:
      strategy (str): model
      dir_experiment_path (str): path of logs that will store images 
      mem_size (int): number of images we want to store with the sample strategy and the specific model architecture
      current_task (int): train task index
      type_strategy (str): sample strategy specified

    Returns:
      Instance of class to apply the specified sample strategy
    '''
    print(f"type_strategy: {type_strategy}")
    if type_strategy=="replay":
        return MemoryReplay(strategy, task_order, dir_experiment_path, mem_size, current_task)

    elif type_strategy=="replay_all_data":
        return MemoryReplayAllData(strategy, task_order, dir_experiment_path, mem_size, current_task)

    elif type_strategy=="cumulative" or type_strategy=="naive" or type_strategy=="multi_task" or type_strategy=="single_model":#without repetition
        return MemoryNaive(strategy, task_order, dir_experiment_path, mem_size, current_task)          

    elif type_strategy=="cl":        
        return MemoryNaive(strategy, task_order, dir_experiment_path, mem_size, current_task)          

    elif type_strategy=="latent_replay":
        return MemoryLatentReplay(strategy, task_order, dir_experiment_path, mem_size, current_task)

    elif type_strategy=="generative_replay":
        return MemoryGenerativeReplay(strategy, task_order, dir_experiment_path, mem_size, current_task)

    elif type_strategy=="compressed_degenerative_replay":
        return MemoryCompressedDegenerativeReplay(strategy, task_order, dir_experiment_path, mem_size, current_task)

    elif type_strategy=="compressed_replay_paste":
        return MemoryCompressedReplayPaste(strategy, task_order, dir_experiment_path, mem_size, current_task)
    else:
        raise ValueError(f"Memory Not defined ! Memory for {type_strategy} doesn't exist !")

# abstract create_memory_for_each_task e memory_update
class Memory:
    """
    It creates memory for each task with the sample strategy specified

    Attributes:
      filepaths (list): paths of images (.png) and their remaining information (.pickle)
      tasks_memory (dict): dictionary containing pairs keys-values, where keys correspond to the task_id and the values are the instance of MemoryDataset class
      #keys represent the task_index during execution
      num_tasks (int): current_task + 1
      current_task (int): training task index

    """
    def __init__(self, strategy, task_order, dir_experiment_path, mem_size, current_task, load_all_tasks = False):
        self.strategy = strategy
        self.task_order = task_order
        self.mem_size = mem_size
        self.dir_experiment_path = dir_experiment_path
        self.filepaths = []
        self.tasks_memory = {}  #stored samples contained into tasks_memory
        #each sample contains image itself (sample[0]), and corresponding info (sample[1], sample[2]...)

        self.load_all_tasks = load_all_tasks

        self.num_tasks = current_task + 1
        
        self.current_task = current_task

        self.create_memory_for_each_task()

    def get_samples_from_memory(self, memory_batch_size, sample_strategy):
        """
        Selects task ids randomly with size (memory_batch_size,) and access to the stored samples contained into tasks_memory

        Args:
          memory_batch_size (int): batch size of the dataloder
          sample_strategy:

        Returns:
          tuple of arrays (tensor,ndarray,ndarray,ndarray,ndarray): (image, object_label, index, anomaly_label, filepath)
        """
        tasks = [i for i in range(self.num_tasks-1)]
        # if you have current_task=3 then tasks=[0,1,2], and if batch_size=8 then random_tasks_ids=[22112002]

        # np.random.seed(self.strategy.seed)
        random_task_ids = np.random.choice(tasks, size=memory_batch_size, replace=True)

        batch_old0, batch_old1, batch_old2, batch_old3, batch_old4 = [], [], [], [], []  # lists respectively: imgs,classes([0..9]),idx,anomaly_info,filepaths

        if self.strategy.parameters['architecture'] == "draem":
            batch_old0 = {}

        for task_id in random_task_ids:
            if len(self.tasks_memory[task_id]) > 0:
                # idx represents the ordinal number of the sample saved for specific task
                # which is executed earlier, which means it ranges from 0 to num_saved_samples for that task
                idx = np.random.randint(0, len(self.tasks_memory[task_id]), 1)[0]

                # pick up the sample corresponding to task_id of specified idx
                sample = self.tasks_memory[task_id][idx]
                if self.strategy.parameters['architecture'] == 'efficientad':
                    batch_old0.append(sample[0].reshape(1,2,3,self.strategy.parameters['crp_size'],self.strategy.parameters['crp_size']))

                elif self.strategy.parameters['architecture'] == 'draem':
                    for key, tensor in sample[0].items():
                        if key == 'image' or key == 'augmented_image':
                            if key not in batch_old0:
                                batch_old0[key] = []
                            batch_old0[key].append(tensor.reshape(1, 3, self.strategy.parameters['crp_size'], self.strategy.parameters['crp_size']))
                        elif key == 'anomaly_mask':
                            if key not in batch_old0:
                                batch_old0[key] = []
                            batch_old0[key].append(tensor.reshape(1, 1, self.strategy.parameters['crp_size'], self.strategy.parameters['crp_size']))
                        elif key == 'has_anomaly':
                            if key not in batch_old0:
                                batch_old0[key] = []
                            batch_old0[key].append(tensor.reshape(1, 1))
                else:
                    if sample_strategy == "compressed_replay_paste":
                        x = sample[0].to(self.strategy.device)
                        if x.dim() == 3:
                            x = x.unsqueeze(0)
                        batch_old0.append(x)
                    else:
                        batch_old0.append(sample[0].reshape(1,3,self.strategy.parameters['crp_size'],self.strategy.parameters['crp_size']))
               
                batch_old1.append(sample[1])      
                batch_old2.append(sample[2])
                batch_old3.append(sample[3])
                batch_old4.append(sample[4])

        if self.strategy.parameters['architecture'] == "draem":
            for key in batch_old0.keys():
                batch_old0[key] = torch.cat(batch_old0[key], dim=0)
        else:
            batch_old0 = torch.cat(batch_old0)

        batch_old1 = torch.from_numpy(np.asarray(batch_old1))
        batch_old2 = np.array(batch_old2) 
        batch_old3 = np.array(batch_old3) 
        batch_old4 = np.array(batch_old4) 
        return batch_old0, batch_old1, batch_old2, batch_old3, batch_old4

    def create_batch_data(self, batch_data, memory_batch_size, sample_strategy):#combines batch from memory(previous tasks) and batch from current task
        """
        Merge the batches corresponding to the current task to the batches of the previous tasks

        Args:
          batch_data: batch of the current task
          memory_batch_size: batch size of the dataloader
          sample_strategy:

        Returns:
           tuple if task index > 0 and sample_strategy is different from naive
           batch_data otherwise
        """

        if (self.strategy.index_training>0 and
                self.strategy.sample_strategy not in ["cl", "naive", "cumulative", "multi_task", "single_model"]):
            batch_old_samples = self.get_samples_from_memory(memory_batch_size, sample_strategy)

            if self.strategy.parameters['architecture'] == "draem":
                batch_old_samples_0 = {}
                for key in batch_old_samples[0].keys():
                    batch_old_samples_0[key] = torch.cat([batch_old_samples[0][key], batch_data[0][key]])
            else:
                if sample_strategy == "compressed_replay_paste":
                    _, bootstrap_feat = self.strategy.trainer.ad_model.teacher(batch_data[0].to(self.strategy.device))
                    batch_old_samples_0 = torch.cat([batch_old_samples[0], bootstrap_feat])
                else:
                    # size: 2x memory batch size
                    batch_old_samples_0 = torch.cat([batch_old_samples[0], batch_data[0]])
             
            batch_old_samples_1 = torch.cat((batch_old_samples[1],batch_data[1]))

            batch_old_samples_2 = torch.cat((torch.tensor(batch_old_samples[2]),batch_data[2]))
            batch_old_samples_3 = np.concatenate((batch_old_samples[3],batch_data[3]))
            batch_old_samples_4 = np.concatenate((batch_old_samples[4],batch_data[4]))

            #shuffle
            dim=0
            if self.strategy.parameters['architecture'] == "draem":
                idx = torch.randperm(batch_old_samples_0["image"].shape[dim])
                for key in batch_old_samples_0.keys():
                    batch_old_samples_0[key] = batch_old_samples_0[key][idx]
            else:
                idx = torch.randperm(batch_old_samples_0.shape[dim])
                batch_old_samples_0 = batch_old_samples_0[idx]

            batch_old_samples_1 = batch_old_samples_1[idx]
            batch_old_samples_2 = batch_old_samples_2[idx]
            batch_old_samples_3 = batch_old_samples_3[idx]
            batch_old_samples_4 = batch_old_samples_4[idx]
            
            return batch_old_samples_0, batch_old_samples_1, batch_old_samples_2, batch_old_samples_3, batch_old_samples_4
        else:
            return batch_data


def change_filename_diz(dir_path,filename_diz):
    filepath_diz = os.path.join(dir_path, filename_diz).replace('\\','/')
    return filepath_diz

def change_filename_img(dir_path,filename_diz):
    filename = filename_diz.replace(".pickle","")
    filename_img = filename + ".png"
    filepath_img = os.path.join(dir_path,filename_img).replace('\\','/')
    return filepath_img

class MemoryFromFolder(Memory):
    def __init__(self,folder_name, strategy, task_order, dir_experiment_path, mem_size, current_task, load_all_tasks=False):
        self.folder_name = folder_name
        super().__init__(strategy, task_order, dir_experiment_path, mem_size, current_task, load_all_tasks)
        #memory = MemoryFromFolder(type_memory,self, task_order, memory_dataset_path, 1000, current_task, load_all_tasks)
        #/data/original_dataset/memorized = /dir_experiment_path/folder_name

    def create_memory_for_each_task(self):
        if self.load_all_tasks is False:
            class_ids = self.task_order[0:self.num_tasks-1]  # it does NOT take into account current task
        else:
            class_ids = self.task_order[0:10]  # takes ALL CLASSES, task_order = [4 5 6 1 2 3 7 8 9 0]

        # for each class taken prior to current task it creates tasks_memory[task_id] by loading stored samples
        for task_id, class_id in enumerate(class_ids):
            dir_path = os.path.join(
                self.dir_experiment_path, f"{self.folder_name}", f"T{self.current_task-1}", f"{class_id}"
            ).replace('\\','/')  # /data/original_dataset/memorized/0/class_id0

            filenames = os.listdir(dir_path)
            filenames_diz = [filename for filename in filenames if filename.endswith(".pickle") and "embed" not in filename]
            filepaths = [ (change_filename_diz(dir_path,filename_diz), change_filename_img(dir_path,filename_diz))  for filename_diz in filenames_diz     ]                
            self.tasks_memory[task_id] = MemoryDataset(filepaths, self.strategy)  # task_id == task_index


class MemoryFromMemorized(MemoryFromFolder):
    def __init__(self, strategy, task_order, dir_experiment_path, mem_size, current_task, load_all_tasks=False):
       super().__init__("memorized",strategy, task_order, dir_experiment_path, mem_size, current_task, load_all_tasks)#folder_name = memorized

class MemoryFromGenerated(MemoryFromFolder):
    def __init__(self, strategy, task_order, dir_experiment_path, mem_size, current_task, load_all_tasks=False):
       super().__init__("generated",strategy, task_order, dir_experiment_path, mem_size, current_task, load_all_tasks)#folder_name = generated


class MemoryNaive(Memory):
    def __init__(self, strategy, task_order, dir_experiment_path, mem_size, current_task):
       super().__init__(strategy, task_order, dir_experiment_path, mem_size, current_task)
    
    def create_memory_for_each_task(self):#creates empty tasks_memory
        class_ids = self.task_order[0:self.num_tasks-1]
        for task_id,class_id in enumerate(class_ids):
                filepaths = []
                self.tasks_memory[task_id] = MemoryDataset(filepaths, self.strategy)

    def memory_update(self,dataset_task, task_id, current_task):
        '''
        Updates memory after the model training on each task, where:
            - Creates the folders memorized and generated for the specified sample strategy.
            - store the data into memorized and generated

        Args:
          dataset_task: data corresponding to the task "task_id" 
          task_id (int)
        '''
        sample_indices = [] 
        for _,idx in enumerate(sample_indices):#it does not work here -> empty
            sample = dataset_task[idx]
            diz = from_sample_to_dict(sample)
            class_id = diz["y"]
            sample_id = diz["idx"]

            # Save diz of sample
            path_memorized = os.path.join(self.dir_experiment_path,"memorized",f"T{task_id}",f"{class_id}").replace('\\','/')
            path_generated = os.path.join(self.dir_experiment_path,"generated",f"T{task_id}",f"{class_id}").replace('\\','/')
            os.makedirs(path_memorized,exist_ok=True)
            os.makedirs(path_generated,exist_ok=True)

            filepath_diz = os.path.join(path_memorized,f"{sample_id}.pickle").replace('\\','/')
            save_pickle(diz,filepath_diz)            

            filepath_diz = os.path.join(path_generated,f"{sample_id}.pickle").replace('\\','/')
            save_pickle(diz,filepath_diz)    

            # Take generated image and save into generated
            x = sample[0]
            output = produce_output_given_model_and_sample(self.strategy,sample)
            x_hat = output[0][0]

            # Save images
            filepath_img = os.path.join(path_memorized,f"{sample_id}.png").replace('\\','/')
            save_img(x, filepath_img)  

            filepath_img = os.path.join(path_generated,f"{sample_id}.png").replace('\\','/')
            save_img(x_hat, filepath_img)  


class MemoryReplayAllData(MemoryFromMemorized):
    def __init__(self, strategy, task_order, dir_experiment_path, mem_size, current_task):
       super().__init__(strategy, task_order, dir_experiment_path, mem_size, current_task)

    def memory_update(self,dataset_task, task_id, current_task):
        sample_indices = list(range(0,len(dataset_task)))
        for _,idx in enumerate(sample_indices):
            sample = dataset_task[idx]
            diz = from_sample_to_dict(sample)
            class_id = diz["y"]
            sample_id = diz["idx"]

            # Save diz of sample
            path_memorized = os.path.join(self.dir_experiment_path,"memorized",f"T{task_id}",f"{class_id}").replace('\\','/')
            path_generated = os.path.join(self.dir_experiment_path,"generated",f"T{task_id}",f"{class_id}").replace('\\','/')
            os.makedirs(path_memorized,exist_ok=True)
            os.makedirs(path_generated,exist_ok=True)

            filepath_diz = os.path.join(path_memorized,f"{sample_id}.pickle").replace('\\','/')
            save_pickle(diz,filepath_diz)#save sample pickle for next tasks (replay method)            

            filepath_diz = os.path.join(path_generated,f"{sample_id}.pickle").replace('\\','/')#all data in memorized and generated folders have ordered labels from 0 to len(dataset_task)
            save_pickle(diz,filepath_diz)#save sample pickle for next tasks (replay method) 

            # Take generated image and save into generated
            x = sample[0]
            output = produce_output_given_model_and_sample(self.strategy,sample)
            x_hat = output[0][0]

            # Save images
            filepath_img = os.path.join(path_memorized,f"{sample_id}.png").replace('\\','/')#save original image
            save_img(x, filepath_img)  

            filepath_img = os.path.join(path_generated,f"{sample_id}.png").replace('\\','/')#save generated image
            save_img(x_hat, filepath_img)  


class MemoryReplay(MemoryFromMemorized):
    '''
    Update 
    '''
    def __init__(self, strategy, task_order, dir_experiment_path, mem_size, current_task):
       super().__init__(strategy, task_order, dir_experiment_path, mem_size, current_task)

    def memory_update(self,dataset_task, task_id, current_task):
        num_samples_per_task = self.mem_size//self.num_tasks
        num_samples_per_task = min(num_samples_per_task, len(dataset_task))

        rng = default_rng()
        sample_indices = rng.choice(len(dataset_task), size=num_samples_per_task, replace=False)

        for _,idx in enumerate(sample_indices):
            sample = dataset_task[idx]
            diz = from_sample_to_dict(sample)
            class_id = diz["y"]
            index_dataset = diz["idx"]

            # Save diz of sample
            path_memorized = os.path.join(self.dir_experiment_path,"memorized",f"T{task_id}",f"{class_id}").replace('\\','/')
            #path_generated = os.path.join(self.dir_experiment_path,"generated",f"T{task_id}",f"{class_id}").replace('\\','/')
            os.makedirs(path_memorized,exist_ok=True)
            #os.makedirs(path_generated,exist_ok=True)

            filepath_diz = os.path.join(path_memorized,f"{index_dataset}.pickle").replace('\\','/')
            save_pickle(diz,filepath_diz)            

            #filepath_diz = os.path.join(path_generated,f"{index_dataset}.pickle").replace('\\','/')
            #save_pickle(diz,filepath_diz)    

            # Take generated image and save into generated
            sample_id = index_dataset
            if self.strategy.parameters['architecture'] == 'efficientad':
                x = sample[0][0]   
            elif self.strategy.parameters['architecture'] == 'draem':
                x = sample[0]["image"]#torch.tensor(3,256,256) 
            else:
                x = sample[0]
            '''    
            output = produce_output_given_model_and_sample(self.strategy,sample)
            x_hat = output[0][0]#(256,256)
            if self.strategy.parameters['architecture'] == 'draem':
                x_hat = torch.from_numpy(x_hat)
            '''
            # Save images
            filepath_img = os.path.join(path_memorized,f"{sample_id}.png").replace('\\','/')

            if self.strategy.parameters['architecture'] == 'draem':
                save_img_draem(x, filepath_img) 
            else:
                save_img(x, filepath_img)  

            #filepath_img = os.path.join(path_generated,f"{sample_id}.png").replace('\\','/')
            #save_img(x_hat, filepath_img)  


class MemoryLatentReplay(MemoryFromGenerated):
    def __init__(self, strategy, task_order, dir_experiment_path, mem_size, current_task):
       super().__init__(strategy, task_order, dir_experiment_path, mem_size, current_task)

    def memory_update(self,dataset_task, task_id, current_task):
        num_samples_per_task = self.mem_size//self.num_tasks
        num_samples_per_task = min(num_samples_per_task, len(dataset_task))
        
        rng = default_rng()
        sample_indices = rng.choice(len(dataset_task), size=num_samples_per_task, replace=False)

        for _,idx in enumerate(sample_indices):
            sample = dataset_task[idx]
            diz = from_sample_to_dict(sample)
            class_id = diz["y"]
            sample_id = diz["idx"]

            # Save diz of sample
            path_memorized = os.path.join(self.dir_experiment_path,"memorized",f"T{task_id}",f"{class_id}").replace('\\','/')
            path_generated = os.path.join(self.dir_experiment_path,"generated",f"T{task_id}",f"{class_id}").replace('\\','/')
            os.makedirs(path_memorized,exist_ok=True)
            os.makedirs(path_generated,exist_ok=True)

            filepath_diz = os.path.join(path_memorized,f"{sample_id}.pickle").replace('\\','/')
            save_pickle(diz,filepath_diz)            

            filepath_diz = os.path.join(path_generated,f"{sample_id}.pickle").replace('\\','/')
            save_pickle(diz,filepath_diz)    

            # Take generated image and save into generated
            x = sample[0]
            output = produce_output_given_model_and_sample(self.strategy,sample)
            x_hat = output[0][0]
            z = output[1][0]

            filepath_latent = os.path.join(path_memorized,f"{sample_id}_latent.pickle").replace('\\','/')
            save_pickle(z,filepath_latent)

            # Save images
            filepath_img = os.path.join(path_memorized,f"{sample_id}.png").replace('\\','/')
            save_img(x, filepath_img)  

            filepath_img = os.path.join(path_generated,f"{sample_id}.png").replace('\\','/')
            save_img(x_hat, filepath_img)  

class MemoryGenerativeReplay(MemoryFromGenerated):
    def __init__(self, strategy, task_order, dir_experiment_path, mem_size, current_task):
       super().__init__(strategy, task_order, dir_experiment_path, mem_size, current_task)

    def memory_update(self,dataset_task, task_id, current_task):
        num_samples_per_task = self.mem_size//self.num_tasks
        num_samples_per_task = min(num_samples_per_task, len(dataset_task))

        rng = default_rng()
        sample_indices = rng.choice(len(dataset_task), size=num_samples_per_task, replace=False)
        for _,idx in enumerate(sample_indices):
            sample = dataset_task[idx]
            diz = from_sample_to_dict(sample)
            class_id = diz["y"]
            sample_id = diz["idx"]

            # Save diz of sample
            path_memorized = os.path.join(self.dir_experiment_path,"memorized",f"T{task_id}",f"{class_id}").replace('\\','/')
            path_generated = os.path.join(self.dir_experiment_path,"generated",f"T{task_id}",f"{class_id}").replace('\\','/')
            os.makedirs(path_memorized,exist_ok=True)
            os.makedirs(path_generated,exist_ok=True)

            filepath_diz = os.path.join(path_memorized,f"{sample_id}.pickle").replace('\\','/')
            save_pickle(diz,filepath_diz)            

            filepath_diz = os.path.join(path_generated,f"{sample_id}.pickle").replace('\\','/')
            save_pickle(diz,filepath_diz)    

            # Take generated image and save into generated
            if self.strategy.parameters['architecture'] == 'efficientad':
                x = sample[0][0]   
            else:
                x = sample[0]

            output = produce_output_given_model_from_noise(self.strategy,sample)
            x_hat = output[0][0]
            noise = output[1][0]

            filepath_latent = os.path.join(path_memorized,f"{sample_id}_noise.pickle").replace('\\','/')
            save_pickle(noise,filepath_latent)

            # Save images
            filepath_img = os.path.join(path_memorized,f"{sample_id}.png").replace('\\','/')
            save_img(x, filepath_img)  

            filepath_img = os.path.join(path_generated,f"{sample_id}.png").replace('\\','/')
            save_img(x_hat, filepath_img)  
        
class MemoryCompressedDegenerativeReplay(MemoryFromGenerated):
    def __init__(self, strategy, task_order, dir_experiment_path, mem_size, current_task):
       super().__init__(strategy, task_order, dir_experiment_path, mem_size, current_task)

    def memory_update(self,dataset_task, task_id, current_task):
        num_samples_per_task = self.mem_size//self.num_tasks
        num_samples_per_task = min(num_samples_per_task, len(dataset_task))

        rng = default_rng()
        sample_indices = rng.choice(len(dataset_task), size=num_samples_per_task, replace=False)
        for _,idx in enumerate(sample_indices):
            sample = dataset_task[idx]
            diz = from_sample_to_dict(sample)
            class_id = diz["y"]
            sample_id = diz["idx"]

            # Save diz of sample
            path_memorized = os.path.join(self.dir_experiment_path,"memorized",f"T{task_id}",f"{class_id}").replace('\\','/')
            path_generated = os.path.join(self.dir_experiment_path,"generated",f"T{task_id}",f"{class_id}").replace('\\','/')
            os.makedirs(path_memorized,exist_ok=True)
            os.makedirs(path_generated,exist_ok=True)

            filepath_diz = os.path.join(path_memorized,f"{sample_id}.pickle").replace('\\','/')
            save_pickle(diz,filepath_diz)            

            filepath_diz = os.path.join(path_generated,f"{sample_id}.pickle").replace('\\','/')
            save_pickle(diz,filepath_diz)    

            # Take generated image and save into generated
            x = sample[0]

            #real_A
            h,w = 32,32
            real_B = x #original tensor 3x224x224
            real_A,real_A_32 = produce_scaled_A_from_B(real_B.clone(),h,w)#blured tensor 3x224x224 obtained from 32x32, numpy.array 3x32x32
            real_B = torch.stack([x])
            real_A = torch.stack([real_A])
            real_A_32 = np.stack([real_A_32])
            model = self.strategy.trainer.vae.decoder.pix2pix_model
            fake_B = forward_test(model, real_A, real_B, model.device).cpu()
            x_hat = fake_B[0]

            filepath_latent = os.path.join(path_memorized,f"{sample_id}_scaled.png").replace('\\','/')
            save_img(real_A_32[0],filepath_latent)

            # Save images
            filepath_img = os.path.join(path_memorized,f"{sample_id}.png").replace('\\','/')
            save_img(x, filepath_img)  

            filepath_img = os.path.join(path_generated,f"{sample_id}.png").replace('\\','/')
            save_img(x_hat, filepath_img)    

class MemoryCompressedReplayPaste(MemoryFromGenerated):
    def __init__(self, strategy, task_order, dir_experiment_path, mem_size, current_task):
       super().__init__(strategy, task_order, dir_experiment_path, mem_size, current_task)

    def create_memory_for_each_task(self):
        # Select class IDs for memory loading
        if self.load_all_tasks is False:
            class_ids = self.task_order[0:self.num_tasks-1] # Ignore current task
        else:
            class_ids = self.task_order[0:10]  # takes ALL CLASSES, task_order = [4 5 6 1 2 3 7 8 9 0]

        # for each class taken prior to current task it creates tasks_memory[task_id] by loading stored samples
        for task_id, class_id in enumerate(class_ids):
            # Define the path for the class samples from previous tasks
            dir_path = os.path.join(self.dir_experiment_path, "memorized", f"T{self.current_task-1}", f"{class_id}").replace('\\','/')

            # Check if directory exists before listing files
            if not os.path.exists(dir_path):
                print(f"Directory {dir_path} does not exist. Skipping class {class_id}.")
                continue

            try:
                filenames = os.listdir(dir_path)
                filenames_diz = [filename for filename in filenames if filename.endswith(".pickle")]
                filepaths = [change_filename_diz(dir_path, filename_diz) for filename_diz in filenames_diz]

                sample_strategy = self.strategy.parameters.get("sample_strategy")

                # Save the memory dataset for the task
                if sample_strategy == "compressed_replay_paste":
                    self.tasks_memory[task_id] = MemoryDataset(filepaths, self.strategy, replay_paste=True)
                else:
                    self.tasks_memory[task_id] = MemoryDataset(filepaths, self.strategy, replay_paste=False)

            except Exception as e:
                print(f"Error loading memory from {dir_path}: {e}")

    def memory_update(self, dataset_task, index_training, task_id_old, current_task, compression=''):
       num_samples_per_task = self.mem_size // self.num_tasks
       num_samples_per_task = min(num_samples_per_task, len(dataset_task))

       # Randomly select a subset of indices from the current task's dataset
       rng = default_rng()
       sample_indices = rng.choice(len(dataset_task), size=num_samples_per_task, replace=False)

       for idx in sample_indices:
           sample = dataset_task[idx]
           diz = from_sample_to_dict(sample)
           class_id = diz["y"]
           index_dataset = diz["idx"]

           if current_task:
               input_tensor = sample[0].unsqueeze(0).to(self.strategy.device)
               _, feature_map = self.strategy.trainer.ad_model.teacher(input_tensor)
           else:
               feature_map = sample[0]

           if compression == "pca" and current_task:
               feature_map = feature_map.squeeze(0)
               C, H, W = feature_map.shape
               reshaped_feature_map = feature_map.reshape(C, -1).T.cpu().numpy()  # Reshape the feature map to (H*W, C) for PCA

               # Step 1: Fit PCA with full components to find optimal n_components
               pca = PCA()
               pca.fit(reshaped_feature_map)

               # Step 2: Calculate cumulative explained variance ratio
               cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
               n_components = np.argmax(cumulative_variance >= 0.98) + 1

               print(f"Training index: {index_training}, PCA components selection T{task_id_old} {n_components}/{C}")

               pca_compressed_map = pca.transform(reshaped_feature_map)[:, :n_components]

               diz['x'] = pca_compressed_map
               diz['pca_params'] = {
                   'mean': pca.mean_,
                   'components': pca.components_[:n_components],
                   'original_channels': C,
                   'height': H,
                   'width': W
               }

           elif compression == "scale_quantization" and current_task:
               quantized_map, min_val, scale = quantize_feature_map(feature_map)
               diz['x'] = quantized_map
               diz['quantization_params'] = {'min_val': min_val.item(), 'scale': scale}
           else:
               diz['x'] = feature_map

           path_memorized = os.path.join(
               self.dir_experiment_path, "memorized", f"T{index_training}", f"{class_id}"
           ).replace('\\', '/')
           os.makedirs(path_memorized, exist_ok=True)

           filepath_diz = os.path.join(path_memorized, f"{index_dataset}.pickle").replace('\\', '/')
           save_pickle(diz, filepath_diz)


def load_memory(strategy,memory_dataset_path,type_memory, task_order, current_task, load_all_tasks=False):
    self = strategy
    memory = MemoryFromFolder(type_memory,self, task_order, memory_dataset_path, 1000, current_task, load_all_tasks)
    return memory


def load_dataset_from_memory(strategy,memory_dataset_path,type_memory,task_order,num_tasks):
    new_stream_data = []
    for temp_index_training in range(num_tasks):# from 0 to 9
        current_task = temp_index_training+1  # from 1 to 10
        memory = load_memory(strategy,memory_dataset_path,type_memory, task_order, current_task)
        dataset_task = memory.tasks_memory[temp_index_training]
        new_stream_data.append(dataset_task)
    return new_stream_data, ConcatDataset(new_stream_data)
