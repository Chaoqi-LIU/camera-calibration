import sys
import open3d as o3d
import numpy as np
import pypose as pp
import sklearn
import sklearn.neighbors
import torch
from typing import List, Tuple

from .utils import get_device



def pcd_registration_loss(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    predicted_transform: pp.se3,
    distance_threshold: float = 0.02,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Compute point-to-point distance between source_pcd and target_pcd
    after applying predicted_transform to source_pcd.

    :param source_pcd: Point cloud of captured object.
    :param target_pcd: Point cloud of model object.
    :param predicted_transform: Predicted SE(3)
    :param distance_threshold: Distance threshold for correspondences.
    :param device: Device to use for computation.
    :return: Loss value.
    """
    
    source_points = torch.from_numpy(np.asarray(source_pcd.points)).to(device)
    target_points = torch.from_numpy(np.asarray(target_pcd.points)).to(device)
    
    source_points = pp.Exp(predicted_transform) @ source_points

    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto')
    knn.fit(target_points.detach().cpu().numpy())

    dist, idx = knn.kneighbors(source_points.detach().cpu().numpy())
    idx = idx.reshape(-1)

    to_keep = np.where(dist < distance_threshold)[0]
    return torch.mean(torch.linalg.norm(
        source_points[to_keep] - 
        target_points[idx[to_keep]], 
    dim=1), dim=0)



def optimize_registration(
    source_pcds: List[o3d.geometry.PointCloud],
    target_pcds: List[o3d.geometry.PointCloud],
    initial_transform: np.ndarray,
    distance_threshold: float = 0.02,
    max_iter: int = 100,
    lr: float = 1e-4,
    stop_threshold: float = 0.01,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Optimize SE(3) transform to minimize point-to-point distance between
    source_pcd and target_pcd.

    :param source_pcd: Point cloud of captured object.
    :param target_pcd: Point cloud of model object.
    :param initial_transform: Initial SE(3) guess.
    :param distance_threshold: Distance threshold for correspondences.
    :param max_iter: Maximum number of iterations.
    :param lr: Learning rate.
    :param stop_threshold: Stop optimization when loss is below this value.
    :param use_gpu: Whether to use GPU for optimization.
    :return: Optimized SE(3) transform, minimum loss.
    """

    device = get_device(use_gpu)

    transform = pp.Log(pp.mat2SE3(initial_transform)).to(device)
    transform.requires_grad = True

    optimizer = torch.optim.Adam([transform], lr=lr)

    min_loss = float('inf')
    for i in range(max_iter):
        sys.stdout.write(f"\rrefinement:: iteration: {i+1}/{max_iter}; loss: {min_loss:.6f}")
        sys.stdout.flush()

        # compute loss
        loss: torch.Tensor = 0.
        for source_pcd, target_pcd in zip(source_pcds, target_pcds):
            loss += pcd_registration_loss(
                source_pcd, target_pcd,
                predicted_transform=transform,
                distance_threshold=distance_threshold,
                device=device
            )

        min_loss = min(min_loss, loss.item())

        # early stopping
        if loss.item() < stop_threshold:
            break
        
        # optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print()
    return pp.matrix(pp.Exp(transform)).detach().cpu().numpy(), min_loss
