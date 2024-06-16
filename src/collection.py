import open3d as o3d
from typing import List, Callable, Tuple

from utils import sample_robot_surface_points



def collect_pcds(
    capture_scene_pcd: Callable,
    query_joint_configuration: Callable,
    robot_urdf: str,
    num_points_on_robot: int
) -> Tuple[
    List[o3d.geometry.PointCloud], 
    List[o3d.geometry.PointCloud]
]:
    """
    Collect point clouds of the scene.

    :param capture_scene_pcd: a function that when called, returns the current
    :param query_joint_configuration: a function that when called, 
        returns the current joint configuration
    :param robot_urdf: the path to the URDF file of the robot
    :param num_points_on_robot: the number of points to sample on the robot surface
    :return: a list of observed point clouds, a list of corresponding ground truth
        robot surface point clouds
    """
    observed_pcds = []
    robot_pcds = []
    while True:

        # user feedback
        while (user_str := input(
            "Press 'c' to capture a point cloud, 'q' to quit: "
        ).strip().lower()) not in ['c', 'q']:
            print("Invalid input")

        # quit data collection
        if user_str == 'q':
            break

        # capture point cloud
        observed_pcd = capture_scene_pcd()
        robot_pcd = sample_robot_surface_points(
            robot_urdf=robot_urdf,
            robot_configuration=query_joint_configuration(),
            num_points=num_points_on_robot
        )

        # visualize
        o3d.visualization.draw_geometries([observed_pcd])
        o3d.visualization.draw_geometries([robot_pcd])

        # user feedback
        while (user_str := input(
            "Press 'y' to accept, 'n' to reject: "
        ).strip().lower()) not in ['y', 'n']: 
            print("Invalid input")

        # store if accepted
        if user_str == 'y':
            observed_pcds.append(observed_pcd)
            robot_pcds.append(robot_pcd)

    return observed_pcds, robot_pcds
