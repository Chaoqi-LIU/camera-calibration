import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from calibrator.utils import load_yaml
from calibrator.calibration import calibrate_camera

from camera import Camera



if __name__ == "__main__":
    configuration = load_yaml("example/configurations.yaml")

    # create customized camera object
    camera = Camera(configuration['camera'])

    # create robot controller object, in this case, a UR robot, and just use RTDE
    import rtde_receive
    receiver = rtde_receive.RTDEReceiveInterface(configuration['robot']['ip_address'])

    # user defined scene pcd capture function
    def capture_scene_pcd():
        # take a snapshot
        pcd = camera.capture_pcd()

        # do some customized post-processing
        # e.g. remove background, statistical outlier, etc.
        to_remove = np.asarray(pcd.points)[:, 2] > \
            configuration['data_collection']['cutoff_depth']
        pcd = pcd.select_by_index(np.where(to_remove)[0], invert=True)

        return pcd

    # user defined joint configuration query function
    def query_joint_configuration():
        return receiver.getActualQ()

    X_robot_camera, error = calibrate_camera(
        capture_scene_pcd=capture_scene_pcd,
        query_joint_configuration=query_joint_configuration,
        robot_urdf=configuration['robot']['urdf'],
        num_points_on_robot=configuration['data_collection']
            ['num_points_on_robot'],
        icp_max_correspondence_distance=configuration
            ['initial_registration']['icp_max_correspondence_distance'],
        pcd_registration_influence_distance=configuration
            ['local_refinement']['pcd_registration_influence_distance']
    )

    print(f"X_robot_camera: \n{repr(X_robot_camera)}")
    print(f"Error: {error}")

    camera.close()
