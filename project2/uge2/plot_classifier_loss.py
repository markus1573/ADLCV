import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("classifier.csv")

n_epochs = 30
logs_per_epoch = len(df) // n_epochs
df["Epoch"] = (df.index // logs_per_epoch) + 1
df = df[df["Epoch"] <= n_epochs]  # drop any remainder rows beyond epoch 30
epoch_loss = df.groupby("Epoch")["Value"].mean().reset_index()

plt.figure(figsize=(10, 5))
plt.plot(epoch_loss["Epoch"], epoch_loss["Value"], marker="o", color="steelblue", linewidth=2, label="Avg CrossEntropy")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Classifier Training Loss (per epoch)")
plt.legend()
plt.tight_layout()
plt.savefig("classifier_loss.png", dpi=150)
plt.show()
