import tqdm
import torch
from torch import nn
from TrajectoryDataset import TrajectoryDatasetTrain
import matplotlib.pyplot as plt

def train_model(model, train_dataloader, val_dataloader, device, optimizer, criterion, scheduler, early_stopping_patience=5, save_weights_name="best_model", epochs=100):
    no_improvement = 0
    best_val_loss = float('inf')
    
    for epoch in tqdm.tqdm(range(epochs), desc="Epoch", unit="epoch"):
        # ---- Training ----
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            batch = batch.to(device)
            pred = model(batch.x)
            y = batch.y.view(batch.num_graphs, 60, 2)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
        
        # ---- Validation ----
        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                pred = model(batch.x)
                y = batch.y.view(batch.num_graphs, 60, 2)
                val_loss += criterion(pred, y).item()

                # show MAE and MSE with unnormalized data
                pred = pred * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                y = y * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                val_mae += nn.L1Loss()(pred, y).item()
                val_mse += nn.MSELoss()(pred, y).item()
        
        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        val_mae /= len(val_dataloader)
        val_mse /= len(val_dataloader)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)   
        else:
            scheduler.step()   
        
        tqdm.tqdm.write(f"Epoch {epoch:03d} | Learning rate {optimizer.param_groups[0]['lr']:.6f} | train normalized MSE {train_loss:8.4f} | val normalized MSE {val_loss:8.4f}, | val unnormalized MAE {val_mae:8.4f} | val unnormalized MSE {val_mse:8.4f}")
        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save(model.state_dict(), save_weights_name + ".pt")
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                print("Early stop!")
                break

def visualize_trajectory(train_data: TrajectoryDatasetTrain, model, idx, device='cpu'):
    train_scene = train_data.get_scene(idx)
    model.eval()  # Set to evaluation mode
    model.to(device)
    with torch.no_grad():
        predictions = model(train_scene.x.unsqueeze(0))
        predictions = predictions.to(device)

    predictions = predictions * train_scene.scale.view(-1,1,1) + train_scene.origin.unsqueeze(1)
    train_sample_y = train_scene.y.unsqueeze(0) * train_scene.scale.view(-1,1,1) + train_scene.origin.unsqueeze(1)
    assert predictions.shape == train_sample_y.shape

    x_pred = predictions[0, :, 0]
    y_pred = predictions[0, :, 1]
    x_gt = train_sample_y[0, :, 0]
    y_gt = train_sample_y[0, :, 1]
    
    plt.figure(figsize=(6, 6))
    plt.plot(x_gt, y_gt, label='Ground Truth', color='blue', marker='o')
    plt.plot(x_pred, y_pred, label='Prediction', color='red', linestyle='--', marker='x')
    
    plt.title(f"Scene {idx} Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    