import open3d as o3d
import numpy as np
import pypose as pp
import sklearn
import sklearn.neighbors
import torch
from typing import Optional, List, Tuple



def pcd_registration_loss(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    predicted_transform: pp.se3,
    distance_threshold: Optional[float] = 0.02,
) -> torch.Tensor:
    """
    Compute point-to-point distance between source_pcd and target_pcd
    after applying predicted_transform to source_pcd.

    :param source_pcd: Point cloud of captured object.
    :param target_pcd: Point cloud of model object.
    :param predicted_transform: Predicted SE(3)
    :param distance_threshold: Distance threshold for correspondences.
    :return: Loss value.
    """
    
    source_points = torch.from_numpy(np.asarray(source_pcd.points))
    target_points = torch.from_numpy(np.asarray(target_pcd.points))
    
    source_points = pp.Exp(predicted_transform) @ source_points

    knn = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto')
    knn.fit(target_points.detach().numpy())

    dist, idx = knn.kneighbors(source_points.detach().numpy())
    idx = idx.view(-1)

    to_keep = dist < distance_threshold
    return torch.mean(torch.linalg.norm(
        source_pcd[to_keep] - 
        target_pcd[idx[to_keep]], 
    dim=1), dim=0)



def optimize_registration(
    source_pcds: List[o3d.geometry.PointCloud],
    target_pcds: List[o3d.geometry.PointCloud],
    initial_transform: np.ndarray,
    distance_threshold: Optional[float] = 0.02,
    max_iter: Optional[int] = 1e4,
    lr: Optional[float] = 1e-5,
    stop_threshold: Optional[float] = 0.01,
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
    :return: Optimized SE(3) transform, minimum loss.
    """

    transform = pp.Log(pp.mat2SE3(initial_transform))
    transform.requires_grad = True

    optimizer = torch.optim.Adam([transform], lr=lr)

    min_loss = float('inf')
    for _ in range(max_iter):

        # compute loss
        loss = sum(
            pcd_registration_loss(
                source_pcd, target_pcd,
                predicted_transform=transform,
                distance_threshold=distance_threshold,
            )
            for source_pcd, target_pcd in zip(source_pcds, target_pcds)
        )

        min_loss = min(min_loss, loss.item())

        # early stopping
        if loss.item() < stop_threshold:
            break
        
        # optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return pp.matrix(pp.Exp(transform)).detach().numpy(), min_loss
