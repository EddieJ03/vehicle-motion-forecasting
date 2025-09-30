import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

class TrajectoryDatasetTrain(Dataset):
    def __init__(self, data, scale=10.0, augment=True, center_agent=0, scene=49):
        """
        data: Shape (N, 50, 110, 6) Training data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        augment: Whether to apply data augmentation (only for training)
        center_agent: The agent to center scene around
        scene: The scene to use for the center_agent's position to center all agents in all other scenes
        """
        self.data = data
        self.scale = scale
        self.augment = augment
        self.center_agent = center_agent
        self.scene = scene

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get_scene(idx, self.augment)

    def get_scene(self, idx, augment=False):
        scene = self.data[idx]
        # Getting 50 historical timestamps and 60 future timestamps
        hist = scene[:, :50, :].copy()    # (agents=50, time_seq=50, 6)
        future = torch.tensor(scene[0, 50:, :2].copy(), dtype=torch.float32)  # (60, 2)
        
        # Data augmentation(only for training)
        if augment:
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
                # Rotate the historical trajectory and future trajectory
                hist[..., :2] = hist[..., :2] @ R
                hist[..., 2:4] = hist[..., 2:4] @ R
                future = future @ R
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                future[:, 0] *= -1

        origin = hist[self.center_agent, self.scene, :2].copy()  # (2,)
        hist[..., :2] = hist[..., :2] - origin
        future = future - origin

        # Normalize the historical trajectory and future trajectory
        hist[..., :4] = hist[..., :4] / self.scale
        future = future / self.scale

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            y=future.type(torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )

        return data_item


class TrajectoryDatasetTest(Dataset):
    def __init__(self, data, scale=10.0, center_agent=0, scene=49):
        """
        data: Shape (N, 50, 110, 6) Testing data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        center_agent: The agent to center scene around
        scene: The scene to use for the center_agent's position to center all agents in all other scenes
        """
        self.data = data
        self.scale = scale
        self.center_agent = center_agent
        self.scene = scene

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Testing data only contains historical trajectory
        scene = self.data[idx]  # (50, 50, 6)
        hist = scene.copy()
        
        origin = hist[self.center_agent, self.scene, :2].copy()
        hist[..., :2] = hist[..., :2] - origin
        hist[..., :4] = hist[..., :4] / self.scale

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )
        return data_item

    def generate_submission_predictions(self, model, device, output_file_name='submission'):
        test_loader = DataLoader(self, batch_size=32, shuffle=False,
                         collate_fn=lambda xs: Batch.from_data_list(xs))

        model = model.to(device)
        model.eval()

        pred_list = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred_norm = model(batch.x)
                pred = pred_norm * batch.scale.view(-1,1,1) + batch.origin.unsqueeze(1)
                pred_list.append(pred.cpu().numpy())
                
        pred_list = np.concatenate(pred_list, axis=0)  # (N,60,2)
        pred_output = pred_list.reshape(-1, 2)  # (N*60, 2)
        output_df = pd.DataFrame(pred_output, columns=['x', 'y'])
        output_df.index.name = 'index'
        output_df.to_csv(f"{output_file_name}.csv", index=True)