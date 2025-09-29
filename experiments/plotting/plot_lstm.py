import re
import matplotlib.pyplot as plt
from plot_utils import extract_data

# Extract data
lstm_epochs, lstm_val_mse = extract_data("../lstm/lstm_results.txt")
lstm_att_epochs, lstm_att_val_mse = extract_data("../lstm_attention/lstm_attention_results.txt")

# Create plot
plt.figure(figsize=(12, 6))

if lstm_val_mse:
    plt.plot(lstm_epochs, lstm_val_mse, label="LSTM", color="blue", marker="^", markersize=3, linestyle="-", alpha=0.7, linewidth=2)
    plt.text(lstm_epochs[-1], lstm_val_mse[-1], f'{lstm_val_mse[-1]:.2f}', 
             fontsize=10, ha='left', va='center', color='blue', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.7))
if lstm_att_val_mse:
    plt.plot(lstm_att_epochs, lstm_att_val_mse, label="LSTM + Attention", color="orange", marker="v", markersize=3, linestyle="-", alpha=0.7, linewidth=2)
    plt.text(lstm_att_epochs[-1], lstm_att_val_mse[-1], f'{lstm_att_val_mse[-1]:.2f}', 
             fontsize=10, ha='left', va='center', color='orange', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='orange', alpha=0.7))

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Unnormalized Validation MSE", fontsize=12)
plt.title("LSTM Models - Validation MSE Comparison", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right", fontsize=11)
lstm_max_epoch = max(max(lstm_epochs, default=0), max(lstm_att_epochs, default=0))
if lstm_max_epoch > 0:
    plt.xlim(0, lstm_max_epoch + 1)

plt.tight_layout()
plt.savefig("lstm_val_mse.png", dpi=300, bbox_inches="tight")
plt.show()
