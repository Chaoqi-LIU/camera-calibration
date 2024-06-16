import numpy as np
from src.utils import load_yaml
from src.calibration import calibrate_camera

from camera import Camera



if __name__ == "__main__":
    calibration_configuration = load_yaml("configurations.yaml")

    def capture_scene_pcd():
        camera = Camera(calibration_configuration['camera'])
        pcd = camera.capture_pcd()

        # do some customized post-processing
        # e.g. remove background
        to_remove = pcd.points[:, 2] > \
            calibration_configuration['data_collection']['cutoff_depth']
        pcd = pcd.select_by_index(np.where(to_remove)[0], invert=True)

        camera.close()
        return pcd

    def query_joint_configuration():
        import rtde_receive
        receiver = rtde_receive.RTDEReceiveInterface(
            calibration_configuration['robot']['ip_address'])
        return receiver.getActualQ()

    X_robot_camera, error = calibrate_camera(
        capture_scene_pcd=capture_scene_pcd,
        query_joint_configuration=query_joint_configuration,
        robot_urdf=calibration_configuration['robot']['urdf'],
        num_points_on_robot=calibration_configuration['data_collection']
            ['num_points_on_robot'],
        icp_max_correspondence_distance=calibration_configuration
            ['initial_registration']['icp_max_correspondence_distance'],
        pcd_registration_influence_distance=calibration_configuration
            ['local_refinement']['pcd_registration_influence_distance']
    )

    print(f"X_robot_camera: \n{repr(X_robot_camera)}")
    print(f"Error: {error}")
