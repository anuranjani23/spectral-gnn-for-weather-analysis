# Spectral Graph Neural Networks for Global Weather Pattern Analysis and Prediction Using Spherical Harmonics

This project implements spectral graph neural networks with spherical harmonics for global weather pattern analysis and prediction.

## Project Structure
- `data/`: Raw and processed datasets
- `src/`: Source code for models and utilities
- `notebooks/`: Jupyter notebooks for exploration
- `results/`: Model outputs and visualizations
- `scripts/`: Shell scripts for automation
- `configs/`: Configuration files

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download data: `./scripts/download_data.sh`
3. Preprocess data: `python -m src.data.preprocess`
4. Run training: `python -m src.train`

## References
- Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere
- torch_harmonics library
