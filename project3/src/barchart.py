import matplotlib.pyplot as plt
import numpy as np

# Data extracted from your evaluation
models = ['facebook/dinov2-smal', 'google/vit-base-patc']
base_acc = [0.8280, 0.7540]
tta_acc = [0.8850, 0.8600]

# Graph setup
x = np.arange(len(models))  # Label locations
width = 0.35               # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Create the dual bars
rects1 = ax.bar(x - width/2, base_acc, width, label='Base Accuracy', color='#3498db')
rects2 = ax.bar(x + width/2, tta_acc, width, label='TTA Accuracy', color='#2ecc71')

# Add labels and styling
ax.set_ylabel('Accuracy Score')
ax.set_title('Accuracy Increase with Test-Time Augmentation (TTA)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.0) # Accuracy is typically 0 to 1
ax.legend()

# Function to add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

# Display the plot
plt.savefig("barchart_tta.png")