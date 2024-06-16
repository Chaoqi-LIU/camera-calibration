import os
import open3d as o3d
import yaml
from klampt import WorldModel
from klampt.io import open3d_convert
from klampt.math import se3
import numpy as np
from typing import List, Optional, Sequence



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

    # initialize robot
    world = WorldModel()
    world.readFile(robot_urdf)
    robot = world.robot(0)
    robot.setConfig(robot.configFromDrivers(robot_configuration))

    # distribute points on each link proportional to the surface area
    link_meshes = []
    link_mesh_areas = np.empty(robot.numLinks())
    for link_idx in range(robot.numLinks()):
        link = robot.link(link_idx)
        link_geom = link.geometry()

        # only take care of triangle meshes
        if link_geom.type() != "TriangleMesh":
            continue

        link_mesh = open3d_convert.to_open3d(link_geom.getTriangleMesh())
        link_mesh.transform(np.asarray(se3.homogeneous(link.getTransform())))
        link_meshes.append(link_mesh)
        link_mesh_areas[link_idx] = link_mesh.get_surface_area()

    # sample points on each link
    num_points_per_link = num_points * link_mesh_areas / link_mesh_areas.sum()
    
    pcd = fuse_pcd([
        link_meshes[i].sample_points_uniformly(int(np.ceil(num_points_per_link[i])))
        for i in range(len(link_meshes))
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
