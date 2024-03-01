import multiprocessing
import time
import subprocess
import random

gpus_available = [0,1,2]
seeds = [random.randint(1,10000), random.randint(10000,20000)]
models = ["storig", "draem", "cfa", "eff", "fastflow"]
cl_scenarios = [300, 100, 40]

def launch_training(gpu_id: int, seed: int,  ):
    print(f"Launching training on GPU {gpu_id}")
    # Run the training script as a subprocess
    subprocess.run(["python", "train_script.py", f"--gpu={gpu_id}"])
    print(f"Training on GPU {gpu_id} finished")

def main():
    runs = []

    #define all the combinations
    for model in models:
        for scenario in cl_scenarios:
            for seed in seeds:
                runs.append({'model' : model, 'scenario' : scenario, 'seed' : seed})

    #create a pool of processes (one for each GPU)
    with multiprocessing.Pool(processes=len(gpus_available)) as pool:

        #launch the initial 


    

if __name__ == "__main__": 
    main()