<<<<<<< HEAD
=======

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

import re

def extract_data_attention(filename):
    try:
        with open(filename, "r") as f:
            text = f.read()
        
        # This pattern handles:
        # 1. The progress bar prefix (ignored)
        # 2. Captures epoch number
        # 3. Captures train normalized MSE
        # 4. Captures val normalized MSE (with optional comma)
        # 5. Captures val MAE
        # 6. Captures val MSE
        pattern = (
            r"Epoch:.*?\]Epoch\s+(\d+)\s+\|"
            r".*?train normalized MSE\s+([\d\.e\-]+)\s*\|"
            r".*?val normalized MSE\s+([\d\.e\-]+),?\s*\|"
            r".*?val MAE\s+([\d\.e\-]+)\s*\|"
            r".*?val MSE\s+([\d\.e\-]+)"
        )
        
        matches = re.findall(pattern, text)
        
        if not matches:
            print(f"Warning: No matching data found in file '{filename}'")
            return [], [], [], [], []
        
        # Convert captured groups to proper types
        epochs = [int(m[0]) for m in matches]
        train_nmse = [float(m[1]) for m in matches]
        val_nmse = [float(m[2]) for m in matches]
        val_mae = [float(m[3]) for m in matches]
        val_mse = [float(m[4]) for m in matches]
        
        return epochs, train_nmse, val_nmse, val_mae, val_mse
        
    except FileNotFoundError:
        print(f"Warning: File '{filename}' not found")
        return [], [], [], [], []

# linear_epochs, linear_train_nmse, linearn_val_nmse, linear_val_mae, linear_val_mse = extract_data_attention("training_stats/linear.txt")
# mlp_epochs, mlp_train_mae, mlp_val_mae = extract_data("training_stats/mlp_results.txt")
# cnn_epochs, cnn_train_mae, cnn_val_mae = extract_data("training_stats/simple_cnn_results.txt")
# deep_epochs, deep_train_mae, deep_val_mae = extract_data("training_stats/deep_cnn_results.txt")
# lstm_epochs, lstm_train_nmse, lstm_val_nmse, lstm_val_mae, lstm_val_mse = extract_data_attention("training_stats/lstm.txt")
attention_epochs, attention_train_nmse, attention_val_nmse, attention_val_mae, attention_val_mse = extract_data_attention("training_stats/attention.txt")

max_epoch = max(
    # max(linear_epochs),
    0,
    # max(mlp_epochs, default=0), 
    # max(cnn_epochs, default=0), 
    # max(deep_epochs, default=0),
    # max(lstm_epochs, default=0),
    max(attention_epochs, default=0)
)


plt.figure(figsize=(14, 10))

plt.subplot(1, 1, 1)

# plt.plot(linear_epochs, linear_train_nmse, label="Linear Train Norm MSE", color="blue", marker="o", markersize=4, linestyle="-", alpha=0.7)
# plt.plot(linear_epochs, linearn_val_nmse, label="Linear Val Norm MSE", color="red", marker="o", markersize=4, linestyle="-", alpha=0.7)
# plt.plot(linear_epochs, linear_val_mae, label="Linear Val MAE", color="green", marker="o", markersize=4, linestyle="-", alpha=0.7)
# plt.plot(linear_epochs, linear_val_mse, label="MLP Val MSE", color="purple", marker="o", markersize=4, linestyle="-", alpha=0.7)
# if mlp_train_mae:
#     plt.plot(mlp_epochs, mlp_train_mae, label="MLP Train MAE", color="blue", marker="o", markersize=4, linestyle="-", alpha=0.7)
#     plt.plot(mlp_epochs, mlp_val_mae, label="MLP Val MAE", color="red", marker="o", markersize=4, linestyle="-", alpha=0.7)
# if cnn_train_mae:
    # plt.plot(cnn_epochs, cnn_train_mae, label="CNN Train MAE", color="green", marker="o", markersize=4, linestyle="-", alpha=0.7)
    # plt.plot(cnn_epochs, cnn_val_mae, label="CNN Val MAE", color="purple", marker="o", markersize=4, linestyle="-", alpha=0.7)
# if deep_train_mae:
    # plt.plot(deep_epochs, deep_train_mae, label="Deep CNN Train MAE", color="orange", marker="o", markersize=4, linestyle="-", alpha=0.7)
    # plt.plot(deep_epochs, deep_val_mae, label="Deep CNN Val MAE", color="deeppink", marker="o", markersize=4, linestyle="-", alpha=0.7)
# if lstm_val_mae:
#     plt.plot(lstm_epochs, lstm_train_nmse, label="LSTM Train MSE", color="gold", marker="o", markersize=4, linestyle="-", alpha=0.7)
#     plt.plot(lstm_epochs, lstm_val_nmse, label="LSTM Val MSE", color="saddlebrown", marker="o", markersize=4, linestyle="-", alpha=0.7)
#     plt.plot(lstm_epochs, lstm_val_mae, label="LSTM Val MAE", color="firebrick", marker="o", markersize=4, linestyle="-", alpha=0.7)
if attention_val_mae:
#     plt.plot(attention_epochs, attention_val_mae, label="Attention Val MAE", color="red", marker="x", markersize=4, linestyle="-", alpha=0.7)
    plt.plot(attention_epochs, attention_train_nmse, label="Attention Train MSE", color="red", marker="o", markersize=4, linestyle="-", alpha=0.7)
    plt.plot(attention_epochs, attention_val_nmse, label="Attention Val MSE", color="blue", marker="o", markersize=4, linestyle="-", alpha=0.7)
    plt.plot(attention_epochs, attention_val_mae, label="Attention Val MAE", color="green", marker="o", markersize=4, linestyle="-", alpha=0.7)
    

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Attention")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
plt.xlim(0, max_epoch + 1)

plt.show()
>>>>>>> d5a99af (updated eda and experiment notebooks and added some new graphs)
