import open3d as o3d
import numpy as np
from typing import Optional, Tuple, List

import registration
import refinement
from utils import fuse_pcd



def calibrate_camera(
    observed_pcds: List[o3d.geometry.PointCloud],
    groud_truth_pcds: List[o3d.geometry.PointCloud],
    icp_max_correspondence_distance: Optional[float] = 0.03,
    pcd_registration_influence_distance: Optional[float] = 0.02,
) -> Tuple[np.ndarray, float]:
    """
    Calibrate the camera.

    :param observed_pcds: a list of observed point clouds, in camera frame
    :param groud_truth_pcds: a list of corresponding ground truth
        robot surface point clouds, in robot frame
    :param icp_max_correspondence_distance: maximum correspondence distance
        for ICP registration
    :param pcd_registration_influence_distance: influence distance for
        point cloud registration
    :return: X_Robot_Cam, error
    """

    return refinement.optimize_registration(
        source_pcds=observed_pcds,
        target_pcds=groud_truth_pcds,
        initial_transform=registration.register(
            source_pcd=fuse_pcd(observed_pcds),
            target_pcd=fuse_pcd(groud_truth_pcds),
            max_correspondence_distance=icp_max_correspondence_distance
        ),
        distance_threshold=pcd_registration_influence_distance
    )
