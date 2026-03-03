import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def process_log():
    try:
        with open(Path(__file__).parent / "job_scripts/error.27897981.err", "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    epochs_data = {}
    current_epoch = None

    for line in lines:
        epoch_match = re.search(r"INFO: Starting epoch (\d+):", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs_data[current_epoch] = []
        elif current_epoch is not None:
            mse_match = re.search(r"MSE=([0-9.]+)", line)
            if mse_match:
                epochs_data[current_epoch].append(float(mse_match.group(1)))

    results = []
    for epoch in sorted(epochs_data.keys()):
        mses = epochs_data[epoch]
        if mses:
            avg_mse = sum(mses) / len(mses)
        else:
            avg_mse = None
        results.append({"Epoch": epoch, "Average Loss (MSE)": avg_mse})

    df = pd.DataFrame(results)
    df.to_csv(Path(__file__).parent / "epoch_losses.csv", index=False)
    print(df.to_markdown(index=False))

    if not df.empty:
        # Loss curve (line plot)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['Epoch'], df['Average Loss (MSE)'], marker='o', linewidth=2, markersize=4, color='steelblue')
        ax.set_title('Loss Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Loss (MSE)')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        plt.savefig(Path(__file__).parent / "loss_curve.png")
        plt.show()
        print("Loss curve saved to loss_curve.png")


process_log()