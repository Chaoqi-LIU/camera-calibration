from src.calibration import calibrate_camera
from src.utils import load_yaml

from collection import collect_pcds



if __name__ == "__main__":
    calibration_configuration = load_yaml("configurations.yaml")

    def query_joint_configuration():
        import rtde_receive
        receiver = rtde_receive.RTDEReceiveInterface(
            calibration_configuration['robot']['ip_address'])
        return receiver.getActualQ()
    
    observed_pcds, ground_truth_pcds = collect_pcds(
        configuration=calibration_configuration,
        query_joint_configuration=query_joint_configuration
    )

    X_robot_camera, error = calibrate_camera(
        observed_pcds=observed_pcds,
        groud_truth_pcds=ground_truth_pcds,
        icp_max_correspondence_distance=calibration_configuration
            ['initial_registration']['icp_max_correspondence_distance'],
        pcd_registration_influence_distance=calibration_configuration
            ['local_refinement']['pcd_registration_influence_distance']
    )

    print(repr(X_robot_camera))
