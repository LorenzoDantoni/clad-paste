import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data for the bars
metrics = ['AD Performance', 'Total Memory', 'Training Time']
values_metric1 = [0.49, 0.42, 0.17, 0.32, 0.53, 0.48, 0.58]
values_metric2 = [141.8, 499.8, 1816.6, 448.6, 368.9, 152.6, 459.9]
values_metric3 = [326, 139, 6, 706, 136, 27, 8]

x = np.arange(len([0,1,2,3,4,5,6]))  # the label locations
width = 0.2  # the width of the bars
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# Create the subplots
fig, ax = plt.subplots(1, 3, figsize=(15, 6))



# Plot the bars for each subplot
ax[0].bar(x, values_metric1, width, label='AD Performance', color = colors)
ax[1].bar(x, values_metric2, width, label='Total Memory', color = colors)
ax[2].bar(x, values_metric3, width, label='Training Time', color = colors)
'''
for i in range(3):
    #ax[i].set_xlabel('Bars')
    # Set the white grid style for all subplots
    ax[i].set(style="whitegrid")  
'''
# Set the y-axis to a logarithmic scale


# Customize the plot for each subplot
for i in range(3):
    #ax[i].set_xlabel('Bars')   
    ax[i].grid(True)
    ax[i].set_title('{}'.format(metrics[i]))
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(['EffAD', 'FastFlow', 'PaDiM', 'DRAEM','CFA', 'STFPM', 'PatchCore'])
    
    #ax[i].legend()
    
#Log scale for training time    
ax[2].set_yscale('log')

ax[0].set_ylabel('f1 pixel-level')  
ax[1].set_ylabel('MB')
#ax[2].set_ylabel('Minutes')
ax[2].set_ylabel('Minutes [log scale]')
plt.tight_layout()
plt.savefig("barplot.png")