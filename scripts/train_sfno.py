#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to evaluate weather forecasting models on test set using TemporalSFNO and NCEP data.
"""

import os
import sys
import argparse
import yaml
import logging
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ncep_dataloader import create_ncep_dataloader
from src.model.sfno_model import TemporalSFNO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TemporalSFNO model')
    parser.add_argument('--config', type=str, default='../config.yaml', help='Path to config file')
    parser.add_argument('--sfno_checkpoint', type=str, help='Path to SFNO model checkpoint')
    parser.add_argument('--data_dir', type=str, help='Directory with NCEP data')
    parser.add_argument('--output_dir', type=str, help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, help='Batch size for evaluation')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for evaluation')
    return parser.parse_args()

def load_config(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    for arg in vars(args):
        if getattr(args, arg) is not None and arg != 'config':
            config[arg] = getattr(args, arg)
    return config

def evaluate_model(model, test_loader, device, variables, output_dir):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating TemporalSFNO"):
            inputs = batch['input'].to(device)       # Shape: [B, H, C, T]
            targets = batch['target'].to(device)     # Shape: [B, H, C, T]

            outputs = model(inputs)

            pred = outputs.detach().cpu().numpy()
            tgt = targets.detach().cpu().numpy()

            all_predictions.append(pred)
            all_targets.append(tgt)

    predictions = np.concatenate(all_predictions, axis=0).reshape(-1, len(variables))
    targets = np.concatenate(all_targets, axis=0).reshape(-1, len(variables))

    metrics = []
    for i, var in enumerate(variables):
        var_pred = predictions[:, i]
        var_target = targets[:, i]
        rmse = np.sqrt(mean_squared_error(var_target, var_pred))
        mae = mean_absolute_error(var_target, var_pred)
        correlation = np.corrcoef(var_pred, var_target)[0, 1]
        metrics.append({
            "variable": var,
            "rmse": rmse,
            "mae": mae,
            "correlation": correlation
        })

    overall_rmse = np.sqrt(mean_squared_error(targets, predictions))
    overall_mae = mean_absolute_error(targets, predictions)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, "temporal_sfno_metrics.csv"), index=False)

    np.save(os.path.join(output_dir, "temporal_sfno_predictions.npy"), predictions[:1000])
    np.save(os.path.join(output_dir, "temporal_sfno_targets.npy"), targets[:1000])

    logger.info(f"TemporalSFNO Overall RMSE: {overall_rmse:.6f}")
    logger.info(f"TemporalSFNO Overall MAE: {overall_mae:.6f}")

    return metrics_df, overall_rmse, overall_mae

def main():
    args = parse_args()
    config = load_config(args)
    os.makedirs(config['output_dir'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and config.get('gpu', False) else 'cpu')
    logger.info(f"Using device: {device}")

    _, _, test_loader = create_ncep_dataloader(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        variables=config['variables'],
        history_steps=config['history_steps'],
        forecast_steps=config['forecast_steps'],
        num_workers=config.get('num_workers', 4)
    )
    logger.info(f"Created test dataloader with {len(test_loader.dataset)} samples")

    if config.get('sfno_checkpoint'):
        sfno_checkpoint = torch.load(config['sfno_checkpoint'], map_location=device)
        model = TemporalSFNO(
            in_channels=len(config['variables']),
            hidden_channels=config['hidden_channels'],
            out_channels=len(config['variables']),
            lmax=config['lmax'],
            history_steps=config['history_steps'],
            forecast_steps=config['forecast_steps']
        )
        model.load_state_dict(sfno_checkpoint['model_state_dict'])
        model = model.to(device)
        logger.info(f"Loaded TemporalSFNO model from {config['sfno_checkpoint']}")

        evaluate_model(model, test_loader, device, config['variables'], config['output_dir'])

    else:
        logger.error("No SFNO checkpoint provided. Exiting.")

if __name__ == '__main__':
    main()
