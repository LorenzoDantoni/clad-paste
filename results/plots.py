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

colors_map = {
    "replay-40" : "dodgerblue", 
    "replay-100" : "orange", 
    "replay-300" : "green", 
    "multi_task" : "red", 
    "naive" : "violet", 
    "CL-10000" : "dodgerblue", 
    "CL-20000" : "orange",
    "CL-30000" : "green",
}

# Create a figure and an array of subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

m = 0 

for i in range(2):
    for j in range(3):

        df = pd.read_csv(f"csv/{models[m]}.csv")

        if models[m] == "patch":
            # set columns order
            desired_order = ["task_index", "CL-30000", "CL-20000", "CL-10000", "multi_task", "naive"]
        else: 
            desired_order = ["task_index", "replay-300", "replay-100", "replay-40", "multi_task", "naive"]

        # Reindex the DataFrame with the desired order
        df = df.reindex(columns=desired_order)

        columns = df.columns[1:]

        print(df.head())

        for column in columns:
            # Plot each y column against x
            print(models[m])

            axes[i, j].plot(df['task_index'], df[column], color = colors_map[column], label=strategy_map.get(column, column))
            axes[i, j].set_xlabel('Task')
            axes[i, j].set_ylabel('f1 pixel-level')
            axes[i, j].set_title(model_map[models[m]])

            """
            if models[m] == "patch": 
                axes[i, j].legend(labels = ["CL-10000", "CL-20000", "CL-30000", "Joint Training", "Fine-Tuning"])
            else:
                axes[i, j].legend(labels = ["Replay-40", "Replay-100", "Replay-300", "Joint Training", "Fine-Tuning"])
            """
            axes[i, j].legend()

            axes[i, j].grid(True)
        m+=1

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.savefig(f"plots/all_plots.png")


