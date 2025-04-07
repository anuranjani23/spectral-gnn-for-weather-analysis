import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
import torch_geometric.data as tg_data
import sys
sys.path.append("../../")
from data.preprocess import *
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NCEPDataset(Dataset):
    """
    Dataset for NCEP/NCAR Reanalysis data.
    """
    def __init__(self, data_dir, variables=['air', 'uwnd', 'vwnd', 'slp'], history_steps=3, 
                 forecast_steps=1, time_range=None, transform=None, use_graph=False, target_vars=None):
        self.data_dir = data_dir
        self.variables = variables
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps
        self.transform = transform
        self.use_graph = use_graph
        self.target_vars = target_vars if target_vars else variables
        self.ds = self._load_ncep_data(time_range)
        self.valid_indices = self._calculate_valid_indices()

        if self.use_graph:
            self.edge_index, self.pos = create_graph_from_grid(
                self.ds.lat.values, self.ds.lon.values)
    
    def _load_ncep_data(self, time_range):
        logger.info(f"Loading NCEP data from {self.data_dir}")
        datasets = []
        for var in set(self.variables + self.target_vars):
            file_path = os.path.join(self.data_dir, f"{var}.nc")
            if os.path.exists(file_path):
                ds = xr.open_dataset(file_path)
                if time_range:
                    ds = ds.sel(time=slice(*time_range))
                datasets.append(ds)
            else:
                raise FileNotFoundError(f"Cannot find {var}.nc in {self.data_dir}")
        
        combined_ds = xr.merge(datasets)
        logger.info(f"Loaded data with time range: {combined_ds.time.values[0]} to {combined_ds.time.values[-1]}")
        return combined_ds
    
    def _calculate_valid_indices(self):
        num_times = len(self.ds.time)
        required_steps = self.history_steps + self.forecast_steps
        if num_times < required_steps:
            raise ValueError(f"Not enough time steps. Need at least {required_steps}, got {num_times}")
        return list(range(num_times - required_steps + 1))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        if self.use_graph:
            history_graphs = [ncep_to_graph_data(self.ds, t, self.variables) for t in range(start_idx, start_idx + self.history_steps)]
            target_graphs = [ncep_to_graph_data(self.ds, t, self.target_vars) for t in range(start_idx + self.history_steps, start_idx + self.history_steps + self.forecast_steps)]
            for graph in history_graphs + target_graphs:
                graph.pos = self.pos
            return {'history': history_graphs, 'target': target_graphs}
        
        history_data = torch.tensor(np.stack([
            np.stack([self.ds[var].isel(time=t).values for var in self.variables], axis=0) 
            for t in range(start_idx, start_idx + self.history_steps)], axis=0), dtype=torch.float32)

        target_data = torch.tensor(np.stack([
            np.stack([self.ds[var].isel(time=t).values for var in self.target_vars], axis=0) 
            for t in range(start_idx + self.history_steps, start_idx + self.history_steps + self.forecast_steps)], axis=0), dtype=torch.float32)

        if self.transform:
            history_data, target_data = self.transform(history_data), self.transform(target_data)
        
        return {'history': history_data, 'target': target_data}


def collate_graph_batch(batch):
    """ Custom collate function for batching graph data. """
    return {
        'history': [tg_data.Batch.from_data_list([sample['history'][t] for sample in batch]) for t in range(len(batch[0]['history']))],
        'target': [tg_data.Batch.from_data_list([sample['target'][t] for sample in batch]) for t in range(len(batch[0]['target']))]
    }


def create_ncep_dataloader(data_dir, batch_size=32, variables=['air', 'uwnd', 'vwnd', 'slp'],
                            history_steps=3, forecast_steps=1, time_range=None, transform=None,
                            use_graph=False, num_workers=4, train_ratio=0.8, val_ratio=0.1):
    dataset = NCEPDataset(data_dir, variables, history_steps, forecast_steps, time_range, transform, use_graph)
    train_size, val_size = int(len(dataset) * train_ratio), int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    logger.info(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
    
    loader_args = {'batch_size': batch_size, 'num_workers': num_workers, 'shuffle': True}
    collate_fn = collate_graph_batch if use_graph else None
    
    return (
        DataLoader(train_dataset, collate_fn=collate_fn, **loader_args),
        DataLoader(val_dataset, collate_fn=collate_fn, shuffle=False, **loader_args),
        DataLoader(test_dataset, collate_fn=collate_fn, shuffle=False, **loader_args)
    )


if __name__ == "__main__":
    data_dir = "../../data/ncep"
    dataset = NCEPDataset(data_dir, use_graph=True)
    sample = dataset[0]
    print(f"Sample history graphs: {len(sample['history'])}, nodes={sample['history'][0].x.shape}, edges={sample['history'][0].edge_index.shape}")
    train_loader, val_loader, test_loader = create_ncep_dataloader(data_dir, batch_size=4, use_graph=True)
    batch = next(iter(train_loader))
    print(f"Batch history graphs: {len(batch['history'])}, nodes={batch['history'][0].x.shape}, edges={batch['history'][0].edge_index.shape}")
