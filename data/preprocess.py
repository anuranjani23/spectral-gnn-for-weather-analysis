import numpy as np
import xarray as xr
import torch
from torch_geometric.data import Data
import os
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def load_ncep_data(data_path, variables=['air.2m.gauss.2024', 'uwnd.10m.gauss.2024', 'vwnd.10m.gauss.2024', 'slp.2024']):
    datasets = []
    for var in variables:
        file_path = os.path.join(data_path, f"{var}.nc")
        if os.path.exists(file_path):
            ds = xr.open_dataset(file_path)
            datasets.append(ds)
    
    if not datasets:
        raise ValueError("No NCEP data files found in the specified directory.")
    
    return xr.merge(datasets)


def create_graph_from_grid(lats, lons):
    """Create graph structure from latitude-longitude grid (compute only once)"""
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    lon_grid, lat_grid = np.meshgrid(lons_rad, lats_rad)
    points = np.column_stack([
        np.cos(lat_grid).flatten() * np.cos(lon_grid).flatten(),
        np.cos(lat_grid).flatten() * np.sin(lon_grid).flatten(),
        np.sin(lat_grid).flatten()
    ])
    
    # Using a smaller subset for triangulation if the grid is large
    if len(points) > 10000:
        print(f"Large grid detected ({len(points)} points). Using KNN instead of full Delaunay triangulation.")
        # Use KNN instead of Delaunay for very large grids
        from sklearn.neighbors import kneighbors_graph
        k = min(12, len(points) - 1)  # Choose appropriate k
        A = kneighbors_graph(points, k, mode='connectivity', include_self=False)
        edges = set()
        coo = A.tocoo()
        for i, j in zip(coo.row, coo.col):
            edges.add((min(i, j), max(i, j)))
    else:
        # Use Delaunay for smaller grids
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
    
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    pos = torch.tensor(points, dtype=torch.float)
    
    print(f"Graph created with {pos.shape[0]} nodes and {edge_index.shape[1]} edges")
    return edge_index, pos


def ncep_to_graph_data(ds, time_idx, edge_index, pos, variables=['air', 'uwnd', 'vwnd', 'slp']):
    """Convert NCEP data to graph data, reusing precomputed edge_index and positions"""
    features = []
    for var in variables:
        if var in ds:
            var_data = ds[var].isel(time=time_idx).values.flatten()
            std_dev = np.std(var_data)
            var_data = (var_data - np.mean(var_data)) / std_dev if std_dev > 0 else np.zeros_like(var_data)
            features.append(var_data)
    
    x = torch.tensor(np.column_stack(features), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, pos=pos)


def create_temporal_dataset(ds, variables=['air', 'uwnd', 'vwnd', 'slp'], 
                           history_steps=3, forecast_steps=3, batch_size=10):
    """Create temporal dataset with batching for memory efficiency"""
    dataset = []
    time_len = len(ds.time)
    
    if time_len < history_steps + forecast_steps:
        raise ValueError("Insufficient time steps in dataset for the requested history and forecast steps.")
    
    # Pre-compute graph structure once
    lats, lons = ds.lat.values, ds.lon.values
    edge_index, pos = create_graph_from_grid(lats, lons)
    
    total_samples = time_len - (history_steps + forecast_steps) + 1
    
    # Process in batches
    for batch_start in range(0, total_samples, batch_size):
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_samples-1)//batch_size + 1}")
        batch_end = min(batch_start + batch_size, total_samples)
        
        for i in range(batch_start, batch_end):
            history_graphs = [ncep_to_graph_data(ds, i + h, edge_index, pos, variables) 
                             for h in range(history_steps)]
            target_graphs = [ncep_to_graph_data(ds, i + history_steps + f, edge_index, pos, variables) 
                           for f in range(forecast_steps)]
            
            dataset.append({
                'history': history_graphs, 
                'target': target_graphs, 
                'timestamp': ds.time.values[i + history_steps]
            })
    
    return dataset


def visualize_graph(data, variable_idx=0, title="Graph Visualization", max_edges=1000):
    """Visualize graph with limited edges for performance"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = data.pos[:, 0], data.pos[:, 1], data.pos[:, 2]
    values = data.x[:, variable_idx].numpy()
    
    # Subsample nodes if there are too many
    if len(x) > 2000:
        indices = np.random.choice(len(x), 2000, replace=False)
        x_sample = x[indices]
        y_sample = y[indices]
        z_sample = z[indices]
        values_sample = values[indices]
        scatter = ax.scatter(x_sample, y_sample, z_sample, c=values_sample, cmap='viridis', s=30, alpha=0.7)
    else:
        scatter = ax.scatter(x, y, z, c=values, cmap='viridis', s=30, alpha=0.7)
    
    # Limit the number of edges for visualization
    if data.edge_index.shape[1] > max_edges:
        edge_indices = np.random.choice(data.edge_index.shape[1], max_edges, replace=False)
        for i in edge_indices:
            idx1, idx2 = data.edge_index[0, i], data.edge_index[1, i]
            ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], [z[idx1], z[idx2]], 'k-', alpha=0.1)
    else:
        for i in range(data.edge_index.shape[1]):
            idx1, idx2 = data.edge_index[0, i], data.edge_index[1, i]
            ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], [z[idx1], z[idx2]], 'k-', alpha=0.1)
    
    ax.set_title(title)
    plt.colorbar(scatter)
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Base project path
    base_path = "/Users/anuranjani/Desktop/my_projects/spectral-gnn-for-weather-analysis"

    # Load raw NCEP data
    data_path = os.path.join(base_path, "data/raw")
    print("Loading NCEP data...")
    ds = load_ncep_data(data_path)
    print("Data loaded successfully!")

    # Create graph structure
    print("Creating graph structure (this is done only once)...")
    lats, lons = ds.lat.values, ds.lon.values
    edge_index, pos = create_graph_from_grid(lats, lons)

    # Convert a sample timestep to graph
    print("Converting first time step to graph data...")
    graph = ncep_to_graph_data(ds, time_idx=0, edge_index=edge_index, pos=pos)

    # Save graph visualization
    print("Creating visualization (with limited edges)...")
    notebooks_dir = os.path.join(base_path, "notebooks")
    os.makedirs(notebooks_dir, exist_ok=True)
    fig = visualize_graph(graph, title="NCEP Data on Graph", max_edges=1000)
    fig_path = os.path.join(notebooks_dir, "ncep_graph_visualization.png")
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Saved graph visualization to {fig_path}")

    # Create and save temporal dataset
    print("Creating temporal dataset (processing in batches)...")
    dataset = create_temporal_dataset(ds, history_steps=5, forecast_steps=1, batch_size=10)
    print(f"Created dataset with {len(dataset)} samples")

    processed_dir = os.path.join(base_path, "data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    dataset_path = os.path.join(processed_dir, "ncep_temporal_dataset.pt")
    torch.save(dataset, dataset_path)
    print(f"Saved temporal dataset to {dataset_path}")

    print("Done!")

