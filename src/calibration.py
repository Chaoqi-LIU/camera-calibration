import open3d as o3d
import numpy as np
from typing import Callable, Sequence, Optional, Tuple

from .collection import collect_pcds
from .registration import register
from .refinement import optimize_registration
from .utils import fuse_pcd



def calibrate_camera(
    capture_scene_pcd: Callable[[], o3d.geometry.PointCloud],
    query_joint_configuration: Callable[[], Sequence[float]],
    robot_urdf: str,
    num_points_on_robot: int = 10000,
    init_guess: Optional[np.ndarray] = None,
    iterations: int = 2,
    stop_threshold: float = 0.01,
    icp_max_correspondence_distance: float = 0.03,
    refinement_max_iter: int = 100,
    refinement_step_size: float = 1e-4,
    pcd_registration_influence_distance: float = 0.02,
    use_gpu: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Calibrate the camera.

    :param capture_scene_pcd: a function that when called, returns the current
    :param query_joint_configuration: a function that when called, 
        returns the current joint configuration
    :param robot_urdf: the path to the URDF file of the robot
    :param num_points_on_robot: the number of points to sample on the robot surface
    :param init_guess: initial guess for the calibration
    :param iterations: number of iterations for the calibration
    :param stop_threshold: stop threshold for the calibration
    :param icp_max_correspondence_distance: maximum correspondence distance
        for ICP registration
    :param refinement_max_iter: maximum number of iterations for the refinement
    :param refinement_step_size: step size for the refinement, aka, learning rate
    :param pcd_registration_influence_distance: influence distance for
        point cloud registration
    :param use_gpu: whether to use GPU for the refinement
    :return: X_Robot_Cam, error
    """

    # collect data
    if init_guess is None:
        observed_pcds, robot_pcds = collect_pcds(
            capture_scene_pcd=capture_scene_pcd,
            query_joint_configuration=query_joint_configuration,
            robot_urdf=robot_urdf,
            num_points_on_robot=num_points_on_robot
        )

    best_error = float("inf")
    best_X_robot_camera = init_guess
    
    for i in range(iterations):

        # early stopping
        if best_error < stop_threshold:
            break

        # ICP registration
        # user will manually select correspondences for the first iteration
        X_robot_camera = register(
            source_pcd=fuse_pcd(observed_pcds),
            target_pcd=fuse_pcd(robot_pcds),
            init_guess=best_X_robot_camera,
            max_correspondence_distance=icp_max_correspondence_distance
        )

        # gradient-based local refinement
        X_robot_camera, error = optimize_registration(
            source_pcds=observed_pcds,
            target_pcds=robot_pcds,
            initial_transform=X_robot_camera,
            distance_threshold=pcd_registration_influence_distance,
            max_iter=refinement_max_iter,
            lr=refinement_step_size,
            stop_threshold=stop_threshold,
            use_gpu=use_gpu
        )

        # update
        if error < best_error:
            best_error = error
            best_X_robot_camera = X_robot_camera

        # log
        print(f"Iteration {i+1}: error {error:.6f}")

    return best_X_robot_camera, best_error
