import open3d as o3d
import numpy as np
from typing import List, Dict, Callable, Tuple

from utils import sample_robot_surface_points
from camera import Camera



def collect_pcds(
    configuration: Dict,
    query_joint_configuration: Callable,
) -> Tuple[
    List[o3d.geometry.PointCloud], 
    List[o3d.geometry.PointCloud]
]:
    """
    Collect point clouds of the scene.

    :param configuration: configuration dictionary
    :param query_joint_configuration: a function that when called, 
        returns the current joint configuration
    :return: a list of observed point clouds, a list of corresponding ground truth
        robot surface point clouds
    """
    camera = Camera(configuration['camera'])

    observed_pcds = []
    robot_pcds = []
    while True:

        # user feedback
        user_str = input("Press 'c' to capture a point cloud, 'q' to quit: ")
        user_str = user_str.strip().lower()
        if user_str == 'q':
            break
        elif user_str != 'c':
            print("Invalid input")
            continue

        # capture a point cloud and mask out unwanted points
        pcd = camera.capture_pcd()
        to_ignore = pcd.points[:, 2] > configuration['data_collection']['cutoff_depth']
        pcd = pcd.select_by_index(np.where(~to_ignore)[0])
    
        observed_pcds.append(pcd)
        robot_pcds.append(sample_robot_surface_points(
            robot_urdf=configuration['robot']['urdf'],
            robot_configuration=query_joint_configuration(),
            num_points=configuration['data_collection']['num_points_on_robot']
        ))

    return observed_pcds, robot_pcds
