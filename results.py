import matplotlib.pyplot as plt
import re

def extract_val_mae_deep(filename):
    epochs = []
    val_maes = []
    
    with open(filename, 'r') as f:
        content = f.read()
        # Use regex to find all validation MAE values
        matches = re.findall(r'Epoch \d+.*?Validation MAE: (\d+\.\d+)', content)
        
        for i, mae in enumerate(matches):
            epochs.append(i+1)
            val_maes.append(float(mae))
            
    return epochs, val_maes

def extract_val_mae_lstm(filename):
    epochs = []
    val_maes = []
    
    with open(filename, 'r') as f:
        content = f.read()
        # Use regex to find all validation MAE values
        matches = re.findall(r'Epoch.*?val MAE\s+(\d+\.\d+)', content)
        
        for i, mae in enumerate(matches):
            epochs.append(i+1)
            val_maes.append(float(mae))
            
    return epochs, val_maes

# Create the plot
plt.figure(figsize=(12, 6))

# Plot validation MAE for both models
epochs_cnn, val_maes_cnn = extract_val_mae_deep('deep_cnn_results.txt')
epochs_lstm, val_maes_lstm = extract_val_mae_lstm('lstm_attention_results.txt')

plt.plot(epochs_cnn, val_maes_cnn, label='CNN Model', color='blue')
plt.plot(epochs_lstm, val_maes_lstm, label='LSTM+Attention Model', color='red')

# Add labels for final values
plt.annotate(f'Final: {val_maes_cnn[-1]:.2f}', 
            xy=(epochs_cnn[-1], val_maes_cnn[-1]), 
            xytext=(10, 10),
            textcoords='offset points',
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.1),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.annotate(f'Final: {val_maes_lstm[-1]:.2f}', 
            xy=(epochs_lstm[-1], val_maes_lstm[-1]), 
            xytext=(10, -10),
            textcoords='offset points',
            ha='left',
            va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.1),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Customize the plot
plt.xlabel('Epoch')
plt.ylabel('Validation MAE')
plt.title('Validation MAE Comparison')
plt.grid(True)
plt.legend()

# Save or display the plot
plt.savefig('validation_mae_comparison.png')
plt.show()