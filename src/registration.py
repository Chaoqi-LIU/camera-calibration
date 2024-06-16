import numpy as np
import open3d as o3d
from typing import List, Optional



def pick_correspondences(
    pcd: o3d.geometry.PointCloud,
) -> List[int]:
    """
    Pick correspondences between two point clouds.

    :param pcd: Point cloud to pick correspondences from.
    :return: List of indices of correspondences.
    """
    print(
        "\n1) Please pick at least three correspondences using [shift + left click]"
        "\n   Press [shift + right click] to undo point picking"
        "\n2) After picking points, press 'Q' to close the window\n"
    )
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()
    


def register(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    max_correspondence_distance: Optional[float] = 0.03,
) -> np.ndarray:
    """
    ICP algorithm to align captured_pcd to model_pcd.

    :param source_pcd: Point cloud of captured object.
    :param target_pcd: Point cloud of model object.
    :param max_correspondence_distance: Distance threshold for correspondences.
    :return: 4x4 transformation matrix, from captured_pcd to model_pcd.
    """

    # manually select correspondences
    source_points_indices = pick_correspondences(source_pcd)
    target_points_indices = pick_correspondences(target_pcd)
    if (
        (not len(source_points_indices) >= 3) or
        (len(source_points_indices) != len(target_points_indices))
    ): 
        raise RuntimeError(
            "Correspondences not selected correctly. "
            "Please select at least three correspondences."
        )
    
    # construct initial guess
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    init_guess = p2p.compute_transformation(
        source_pcd,
        target_pcd,
        o3d.utility.Vector2iVector(
            np.stack([
                source_points_indices,
                target_points_indices
            ], axis=1)
        )
    )

    # ICP registration
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=max_correspondence_distance,
        init=init_guess,
        estimation_method=o3d.pipelines.registration
            .TransformationEstimationPointToPoint()
    )

    return reg_p2p.transformation
    