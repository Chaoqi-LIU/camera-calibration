import open3d as o3d
import numpy as np
from typing import Optional, Callable, Tuple

import collection
import registration
import refinement
from utils import fuse_pcd



def calibrate_camera(
    capture_scene_pcd: Callable,
    query_joint_configuration: Callable,
    robot_urdf: str,
    num_points_on_robot: int,
    iterations: Optional[int] = 3,
    stop_threshold: Optional[float] = 0.01,
    icp_max_correspondence_distance: Optional[float] = 0.03,
    refinement_max_iter: Optional[int] = 1e4,
    refinement_step_size: Optional[float] = 1e-5,
    pcd_registration_influence_distance: Optional[float] = 0.02,
) -> Tuple[np.ndarray, float]:
    """
    Calibrate the camera.

    :param capture_scene_pcd: a function that when called, returns the current
    :param query_joint_configuration: a function that when called, 
        returns the current joint configuration
    :param robot_urdf: the path to the URDF file of the robot
    :param num_points_on_robot: the number of points to sample on the robot surface
    :param iterations: number of iterations for the calibration
    :param stop_threshold: stop threshold for the calibration
    :param icp_max_correspondence_distance: maximum correspondence distance
        for ICP registration
    :param refinement_max_iter: maximum number of iterations for the refinement
    :param refinement_step_size: step size for the refinement, aka, learning rate
    :param pcd_registration_influence_distance: influence distance for
        point cloud registration
    :return: X_Robot_Cam, error
    """

    # collect data
    observed_pcds, robot_pcds = collection.collect_pcds(
        capture_scene_pcd=capture_scene_pcd,
        query_joint_configuration=query_joint_configuration,
        robot_urdf=robot_urdf,
        num_points_on_robot=num_points_on_robot
    )

    best_error = float("inf")
    best_X_robot_camera = None

    for _ in range(iterations):

        # early stopping
        if best_error < stop_threshold:
            break

        # ICP registration
        # user will manually select correspondences for the first iteration
        X_robot_camera = registration.register(
            source_pcd=fuse_pcd(observed_pcds),
            target_pcd=fuse_pcd(robot_pcds),
            init_guess=best_X_robot_camera,
            max_correspondence_distance=icp_max_correspondence_distance
        )

        # gradient-based local refinement
        X_robot_camera, error = refinement.optimize_registration(
            source_pcds=observed_pcds,
            target_pcds=robot_pcds,
            initial_transform=X_robot_camera,
            distance_threshold=pcd_registration_influence_distance,
            max_iter=refinement_max_iter,
            lr=refinement_step_size,
            stop_threshold=stop_threshold
        )

        # update
        if error < best_error:
            best_error = error
            best_X_robot_camera = X_robot_camera

    return best_X_robot_camera, best_error
