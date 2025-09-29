import matplotlib.pyplot as plt
from plot_utils import extract_data

# Extract data
linreg_epochs, linreg_val_mse = extract_data("../linear_regression/linear_regression_results.txt")

# Create plot
plt.figure(figsize=(12, 6))

if linreg_val_mse:
    plt.plot(linreg_epochs, linreg_val_mse, label="Linear Regression", color="red", marker="o", markersize=3, linestyle="-", alpha=0.7, linewidth=2)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Unnormalized Validation MSE", fontsize=12)
plt.title("Linear Regression - Validation MSE", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right", fontsize=11)
if linreg_epochs:
    plt.xlim(0, max(linreg_epochs) + 1)

plt.tight_layout()
plt.savefig("linear_regression_val_mse.png", dpi=300, bbox_inches="tight")
plt.show()
