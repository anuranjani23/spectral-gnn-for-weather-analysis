import numpy as np
import torch
import torch_geometric
import torch_harmonics as th

def compute_laplacian(edge_index, num_nodes, normalization='sym'):
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float, device=edge_index.device)
    return torch_geometric.utils.get_laplacian(edge_index, edge_weight, normalization=normalization, num_nodes=num_nodes)

def compute_eigen_decomposition(L, k=100):
    L_dense = torch_geometric.utils.to_dense_adj(L[0], edge_attr=L[1]).squeeze(0)
    eigvals, eigvecs = torch.linalg.eigh(L_dense)
    return eigvals[:k], eigvecs[:, :k]

def spectral_graph_conv(x, eigvals, eigvecs, kernel_function):
    x_hat = torch.matmul(eigvecs.T, x)
    kernel_coeffs = kernel_function(eigvals)
    out_hat = x_hat * kernel_coeffs.unsqueeze(1)
    return torch.matmul(eigvecs, out_hat)

def spherical_harmonic_transform(data, lmax=10, nlat=None, nlon=None):
    x, y, z = data.pos[:, 0], data.pos[:, 1], data.pos[:, 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)

    theta_rounded = torch.round(theta * 1e4) / 1e4
    phi_rounded = torch.round(phi * 1e4) / 1e4
    sorted_indices = torch.from_numpy(np.lexsort((phi_rounded.numpy(), theta_rounded.numpy())))

    x_sorted = data.x[sorted_indices]
    if nlat is None or nlon is None:
        raise ValueError("You must specify nlat and nlon")

    features = x_sorted.reshape(nlat, nlon, -1)
    transform = th.RealSHT(nlat, nlon, lmax)
    coeffs = torch.stack([transform(features[:, :, i]) for i in range(features.shape[-1])], dim=-1)

    return coeffs


def inverse_spherical_harmonic_transform(sh_coeffs, nlat, nlon):
    lmax = int(np.sqrt(sh_coeffs.shape[0]) - 1)
    transform = th.RealSHT(nlat, nlon, lmax)
    return torch.stack([transform.inverse(sh_coeffs[..., i]) for i in range(sh_coeffs.shape[-1])], dim=-1)

def chebyshev_polynomial(x, k):
    T = torch.zeros((len(x), k+1), device=x.device)
    T[:, 0] = 1.0
    if k > 0:
        T[:, 1] = x
        for i in range(2, k+1):
            T[:, i] = 2 * x * T[:, i-1] - T[:, i-2]
    return T

if __name__ == "__main__":
    num_nodes = 100
    edge_index = torch.randint(0, num_nodes, (2, 300))
    x = torch.randn(num_nodes, 3)
    L = compute_laplacian(edge_index, num_nodes)
    eigvals, eigvecs = compute_eigen_decomposition(L, k=50)
    kernel_fn = lambda eigvals: torch.exp(-eigvals)
    y = spectral_graph_conv(x, eigvals, eigvecs, kernel_fn)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

