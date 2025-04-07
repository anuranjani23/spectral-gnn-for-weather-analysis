#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train GNN model for weather forecasting using NCEP data.
"""

import os
import sys
import argparse
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ncep_dataloader import create_ncep_dataloader
from src.model.gnn_model import GNNModel  # Replace with actual class name if different

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN model for weather forecasting')
    parser.add_argument('--config', type=str, default='../config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, help='Directory with NCEP data')
    parser.add_argument('--output_dir', type=str, help='Directory to save model and logs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--history_steps', type=int, help='Number of history time steps')
    parser.add_argument('--forecast_steps', type=int, help='Number of forecast time steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def load_config(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    for arg in vars(args):
        if getattr(args, arg) is not None and arg != 'config':
            config[arg] = getattr(args, arg)
    return config

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        history = [g.to(device) for g in batch['history']]
        target = [g.to(device) for g in batch['target']]

        optimizer.zero_grad()
        outputs = model(history)

        loss = 0
        for t in range(len(target)):
            pred = outputs[:, t, :].view(-1, target[t].x.size(1))
            tgt = target[t].x
            loss += criterion(pred, tgt)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            history = [g.to(device) for g in batch['history']]
            target = [g.to(device) for g in batch['target']]

            outputs = model(history)

            loss = 0
            for t in range(len(target)):
                pred = outputs[:, t, :].view(-1, target[t].x.size(1))
                tgt = target[t].x
                loss += criterion(pred, tgt)

            total_loss += loss.item()

    return total_loss / len(val_loader)

def main():
    args = parse_args()
    config = load_config(args)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    os.makedirs(config['output_dir'], exist_ok=True)
    checkpoint_dir = os.path.join(config['output_dir'], 'checkpoints')
    logs_dir = os.path.join(config['output_dir'], 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    config_path = os.path.join(config['output_dir'], 'config.yaml')
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    logger.info(f"Configuration saved to {config_path}")

    device = torch.device('cuda' if torch.cuda.is_available() and config.get('gpu', False) else 'cpu')
    logger.info(f"Using device: {device}")

    # Create dataloaders using NCEP
    train_loader, val_loader, test_loader = create_ncep_dataloader(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        variables=config['variables'],
        history_steps=config['history_steps'],
        forecast_steps=config['forecast_steps'],
        use_graph=True,
        num_workers=config.get('num_workers', 4)
    )
    logger.info(f"Created dataloaders: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val / {len(test_loader.dataset)} test")

    in_channels = len(config['variables'])
    out_channels = len(config['variables'])

    model = GNNModel(  # Replace with actual model class name if different
        in_channels=in_channels,
        hidden_channels=config['hidden_channels'],
        out_channels=out_channels,
        history_steps=config['history_steps'],
        forecast_steps=config['forecast_steps']
    )
    model = model.to(device)
    logger.info(f"Initialized GNN model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 1e-5))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    writer = SummaryWriter(logs_dir)

    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        logger.info(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, os.path.join(checkpoint_dir, 'best_gnn_model.pt'))
            logger.info(f"Saved best model with val_loss: {best_val_loss:.6f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config
        }, os.path.join(checkpoint_dir, 'latest_gnn_model.pt'))

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.6f}")
    writer.close()

if __name__ == '__main__':
    main()
