
import re

def extract_data(filename):
    try:
        with open(filename, "r") as f:
            text = f.read()
        
        pattern = r"Epoch (\d+) \| Learning rate [\d\.]+ \| train normalized MSE\s+[\d\.]+ \| val normalized MSE\s+[\d\.]+, \| val unnormalized MAE\s+[\d\.]+ \| val unnormalized MSE\s+([\d\.]+)"
        matches = re.findall(pattern, text)
        
        if matches:
            epochs = [int(m[0]) for m in matches]
            unnorm_mse = [float(m[1]) for m in matches]
            return epochs, unnorm_mse
        
        print(f"Warning: No matches found in '{filename}'")
        return [], []
    except FileNotFoundError:
        print(f"Warning: File '{filename}' not found")
        return [], []