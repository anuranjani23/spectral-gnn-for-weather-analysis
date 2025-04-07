import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

from src.model.sfno_model import TemporalSFNO  # Update this import path as needed
from src.data.ncep_dataloader import NCEP_Dataset  # Make sure this is implemented

# Metric: Mean Squared Error
def compute_mse(pred, target):
    return ((pred - target) ** 2).mean().item()

def evaluate(model, dataloader, device):
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = batch  # shape: [B, T, C, H, W]
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)  # shape: [B, forecast_steps, C, H, W]
            mse = compute_mse(outputs, targets)
            total_mse += mse * inputs.size(0)

    return total_mse / len(dataloader.dataset)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & Dataloader
    dataset = NCEP_Dataset(
        data_path=args.data_path,
        history_steps=args.history_steps,
        forecast_steps=args.forecast_steps,
        split='test'
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = TemporalSFNO(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        out_channels=args.out_channels,
        lmax=args.lmax,
        nlat=args.nlat,
        nlon=args.nlon,
        history_steps=args.history_steps,
        forecast_steps=args.forecast_steps,
        inference_mode=True
    ).to(device)

    # Load checkpoint
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded model from {args.checkpoint}")
    else:
        print("No checkpoint provided. Evaluating randomly initialized model.")

    # Run evaluation
    mse = evaluate(model, dataloader, device)
    print(f"Test MSE: {mse:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to NCEP dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--out_channels', type=int, default=4)
    parser.add_argument('--lmax', type=int, default=20)
    parser.add_argument('--nlat', type=int, default=32)
    parser.add_argument('--nlon', type=int, default=64)
    parser.add_argument('--history_steps', type=int, default=3)
    parser.add_argument('--forecast_steps', type=int, default=1)

    args = parser.parse_args()
    main(args)
