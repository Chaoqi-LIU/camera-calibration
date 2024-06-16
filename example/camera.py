import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from typing import Tuple, Dict



class Camera:
    """
    Wrapper for operations of Overhead RealSense L515
    """

    def __init__(self, configuration: Dict):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(configuration['ip_address'])

        # #TODO: decide a better resolution for both and perhaps align?
        config.enable_stream(
            rs.stream.depth, 
            configuration['depth_stream']['width'],
            configuration['depth_stream']['height'],
            rs.format.z16, 
            configuration['depth_stream']['fps']
        )
        config.enable_stream(
            rs.stream.color,
            configuration['color_stream']['width'],
            configuration['color_stream']['height'],
            rs.format.bgr8, 
            configuration['color_stream']['fps']
        )
        self.numpix = configuration['depth_stream']['width'] \
                    * configuration['depth_stream']['height']

        print(f"starting ... ")
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        rgb_sensor = profile.get_device().query_sensors()[1]
        #depth_sensor.set_option(rs.option.min_distance, 0)
        ## setting presets
        extra_settings_dict = {
            'laser_power':rs.option.laser_power,
            'min_distance':rs.option.min_distance,
            'receiver_gain':rs.option.receiver_gain,
            'noise_filtering':rs.option.noise_filtering,
            'pre_processing_sharpening':rs.option.pre_processing_sharpening,
            'post_processing_sharpening':rs.option.post_processing_sharpening,
            'confidence_threshold':rs.option.confidence_threshold
        }
        extra_settings = {
            "laser_power" : 0,
            "min_distance" : 0.0,
            "receiver_gain" : 18.0,
            "noise_filtering" : 2.0,
            "pre_processing_sharpening" : 2.0
        }
        for key in extra_settings.keys():
            prop = extra_settings_dict[key]
            depth_sensor.set_option(prop,extra_settings[key])

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.intrinsics = None
        self.inv_intrinsics = None
        self.depth_scale = depth_sensor.get_depth_scale()
        self.update_intrinsics()

        print("Waiting for auto-exposure adjustment...")
        for t in range(5):
            self.pipeline.wait_for_frames()
        print('L515 activated with specified parameters')


    def update_intrinsics(self):
        """
        Capture frames and update camera intrinsics in case of change of resolution, etc.
        """
        while True:

            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            rgb_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not rgb_frame:
                continue

            depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            rgb_intrinsics = rgb_frame.profile.as_video_stream_profile().intrinsics
            self.intrinsics = depth_intrinsics
            break

        # convert rs2 intrinsics to 3x3 matrix
        self.intrinsics = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ])
        self.inv_intrinsics = np.linalg.inv(self.intrinsics)


    def capture_pcd(self) -> o3d.geometry.PointCloud:
        """
        Capture pointcloud data and return it as an open3d pointcloud object
        """
        depth, rgb = self.capture_rgbd()

        H, W = depth.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x = x.reshape(-1)
        y = y.reshape(-1)
        depth = depth.reshape(-1)
        rgb = rgb.reshape(-1, 3)
        valid_depth = depth > 0
        x = x[valid_depth]
        y = y[valid_depth]
        depth = depth[valid_depth]
        rgb = rgb[valid_depth]
        points = self.pix2cam(np.stack([x, y], axis=1), depth)

        # reject points with z <= 0
        reject_mask = points[:, 2] <= 0
        points = points[~reject_mask]
        rgb = rgb[~reject_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

        return pcd


    def capture_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture RGBD data and return it as a numpy arrays
        """
        while True:

            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            rgb_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not rgb_frame:
                continue

            depth_data = np.asanyarray(aligned_depth_frame.get_data())*self.depth_scale
            rgb_data = np.asanyarray(rgb_frame.get_data())

            return depth_data, rgb_data


    def close(self):
        self.pipeline.stop()
        print("L515 deactivated")
    
    
    def cam2pix(self, points: np.ndarray) -> np.ndarray:
        """
        Convert points from camera frame to pixel frame
        """
        # points: (N, 3), points in camera frame
        # return points: (N, 2), points in pixel frame
        points = points @ self.intrinsics.T
        points = points[:, :2] / points[:, 2:]
        return points
    

    def pix2cam(self, points: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Convert points from pixel frame to camera frame
        """
        # points: (N, 2), points in pixel frame
        # depth: (N,), depth values
        # return points: (N, 3), points in camera frame
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points = points * depth[:, None]
        return points @ self.inv_intrinsics.T
    