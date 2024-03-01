import pandas as pd
import matplotlib.pyplot as plt

models = ["cfa" , "storig", "eff", "patch", "draem", "fastflow"]

model_map = {
    "cfa" : "CFA", 
    "storig" : "STPFM", 
    "eff" : "EfficientAD", 
    "draem" : "DRAEM", 
    "patch" : "PatchCore", 
    "fastflow" : "FastFlow",
    "padim" : "Padim" 
}

strategy_map = {
    "naive" : "Fine-Tuning", 
    "multi_task" : "Joint Training", 
    "single_model" : "Single Model", 
    "replay" : "Replay",
    "cl" : "CL", 
    "replay-40" : "Replay-40",
    "replay-100" : "Replay-100",
    "replay-300" : "Replay-300",
}

# Create a figure and an array of subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

m = 0 

for i in range(2):
    for j in range(3):

        df = pd.read_csv(f"csv/{models[m]}.csv")

        columns = df.columns[1:]

        for column in columns:
            if column != "single_model":
                # Plot each y column against x
                axes[i, j].plot(df['task_index'], df[column], label=strategy_map.get(column, column))
                axes[i, j].set_xlabel('Task')
                axes[i, j].set_ylabel('f1 pixel-level')
                axes[i, j].set_title(model_map[models[m]])

                axes[i, j].legend()
                axes[i, j].grid(True)
        m+=1

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.savefig(f"plots/all_plots.png")


