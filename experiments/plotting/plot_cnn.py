import matplotlib.pyplot as plt
from plot_utils import extract_data

# Extract data
cnn_epochs, cnn_val_mse = extract_data("../cnn/simple_cnn/simple_cnn_results.txt")
deep_epochs, deep_val_mse = extract_data("../cnn/deep_cnn/deep_cnn_results.txt")

# Create plot
plt.figure(figsize=(12, 6))

if cnn_val_mse:
    plt.plot(cnn_epochs, cnn_val_mse, label="Simple CNN", color="green", marker="s", markersize=3, linestyle="-", alpha=0.7, linewidth=2)
if deep_val_mse:
    plt.plot(deep_epochs, deep_val_mse, label="Deep CNN", color="purple", marker="d", markersize=3, linestyle="-", alpha=0.7, linewidth=2)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Unnormalized Validation MSE", fontsize=12)
plt.title("CNN Models - Validation MSE Comparison", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right", fontsize=11)
cnn_max_epoch = max(max(cnn_epochs, default=0), max(deep_epochs, default=0))
if cnn_max_epoch > 0:
    plt.xlim(0, cnn_max_epoch + 1)
plt.ylim(bottom=0, top=200)
plt.tight_layout()
plt.savefig("cnn_val_mse.png", dpi=300, bbox_inches="tight")
plt.show()
