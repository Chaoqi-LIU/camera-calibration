import os
import torch
import open3d as o3d
import yaml
from klampt import WorldModel
from klampt.io import open3d_convert
from klampt.math import se3
import numpy as np
from typing import List, Optional, Sequence



def get_device(use_gpu: bool) -> torch.device:
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')



def fuse_pcd(
    pcds: List[o3d.geometry.PointCloud],
) -> o3d.geometry.PointCloud:
    """
    Fuse multiple point clouds into a single point cloud.

    :param pcds: List of point clouds to fuse.
    :return: Fused point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    for p in pcds:
        if p is not None:
            pcd += p
    return pcd



def load_pcds(
    pcd_dir: str,
) -> List[o3d.geometry.PointCloud]:
    """
    Load point clouds from a directory

    :param pcd_dir: Directory containing point clouds.
    :return: List of point clouds.
    """
    pcds = []
    for f in os.listdir(pcd_dir):
        if f.endswith(".ply"):
            pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, f))
            pcds.append(pcd)
    return pcds



def save_pcds(
    pcds: List[o3d.geometry.PointCloud],
) -> None:
    """
    Save point clouds to a directory

    :param pcds: List of point clouds to save.
    """
    for i, p in enumerate(pcds):
        o3d.io.write_point_cloud(f"pcd_{i}.ply", p)



def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)



def sample_robot_surface_points(
    robot_urdf: str,
    robot_configuration: Sequence[float],
    num_points: Optional[int] = 10000,
) -> o3d.geometry.PointCloud:
    """
    Uniformly sample points on the surface of a robot.

    :param robot_urdf: URDF file of the robot.
    :param robot_configuration: Configuration of the robot.
    :param num_points: Number of points to sample.
    """
    if not os.path.exists(robot_urdf):
        raise FileNotFoundError(f"Robot description not found: {robot_urdf}")

    # initialize robot
    world = WorldModel()
    world.readFile(robot_urdf)
    robot = world.robot(0)
    robot.setConfig(robot.configFromDrivers(robot_configuration))

    # distribute points on each link proportional to the surface area
    link_meshes = []
    link_mesh_areas = []
    for link_idx in range(robot.numLinks()):
        link = robot.link(link_idx)
        link_geom = link.geometry()

        # only take care of triangle meshes
        if link_geom.type() != "TriangleMesh":
            continue

        link_mesh = open3d_convert.to_open3d(link_geom.getTriangleMesh())
        link_mesh.transform(np.asarray(se3.homogeneous(link.getTransform())))
        link_meshes.append(link_mesh)
        link_mesh_areas.append(link_mesh.get_surface_area())
        if link_mesh_areas[-1] <= 0:
            raise ValueError(f"Link {link_idx} has nonpositive surface area")

    # sample points on each link
    link_mesh_areas = np.asarray(link_mesh_areas)
    num_points_per_link = num_points * link_mesh_areas / link_mesh_areas.sum()
    num_points_per_link = np.ceil(num_points_per_link).astype(int)

    pcd = fuse_pcd([
        mesh.sample_points_uniformly(nsample) if nsample > 0 else None 
        for (mesh, nsample) in zip(link_meshes, num_points_per_link)
    ])
    
    # remove overhead points
    pcd = pcd.select_by_index(
        np.random.choice(
            len(pcd.points), 
            len(pcd.points) - num_points, 
            replace=False
        ), invert=True
    )
    return pcd
