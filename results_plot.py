
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_data(filename):
    try:
        with open(filename, "r") as f:
            text = f.read()
        
        pattern = r"Epoch (\d+), Train Loss: [\d\.]+, Train MAE: ([\d\.]+), Validation Loss: [\d\.]+, Validation MAE: ([\d\.]+)"
        matches = re.findall(pattern, text)
        
        epochs = [int(m[0]) for m in matches]
        train_mae = [float(m[1]) for m in matches]
        val_mae = [float(m[2]) for m in matches]
        
        return epochs, train_mae, val_mae
    except FileNotFoundError:
        print(f"Warning: File '{filename}' not found")
        return [], [], []

mlp_epochs, mlp_train_mae, mlp_val_mae = extract_data("training_results/mlp_results.txt")
cnn_epochs, cnn_train_mae, cnn_val_mae = extract_data("training_results/simple_cnn_results.txt")
deep_epochs, deep_train_mae, deep_val_mae = extract_data("training_results/deep_cnn_results.txt")

max_epoch = max(
    max(mlp_epochs, default=0), 
    max(cnn_epochs, default=0), 
    max(deep_epochs, default=0)
)

plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
if mlp_train_mae:
    plt.plot(mlp_epochs, mlp_train_mae, label="MLP Train MAE", color="blue", marker="o", markersize=4, linestyle="-", alpha=0.7)
if cnn_train_mae:
    plt.plot(cnn_epochs, cnn_train_mae, label="CNN Train MAE", color="green", marker="s", markersize=4, linestyle="-", alpha=0.7)
if deep_train_mae:
    plt.plot(deep_epochs, deep_train_mae, label="Deep Model Train MAE", color="purple", marker="d", markersize=4, linestyle="-", alpha=0.7)

plt.xlabel("Epoch")
plt.ylabel("Train MAE")
plt.title("Training MAE Comparison")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.xlim(0, max_epoch + 1)

plt.show()
