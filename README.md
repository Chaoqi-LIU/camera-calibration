# Toolkit for camera calibration

## ICP-based camera calibration

### What's the essential idea behind?
We capture point clouds (PCDs) of the robot in the scene, we uniformly sample PCDs on the robot surface, and we run Iterative Closest Point (ICP) to align observed robot PCDs and the 'ground truth' robot PCDs. The script will then produce the camera pose in robot frame, aka, rigid body transformation from camera frame to robot frame. Again, not camera pose in world frame but in robot frame!

### How to use the script?
Simply follow the process as shown in `/example`. Details about the function arguments can be found in the corresponding function definition. At minimal, user only needs to provide one function that takes nothing as input and return the desired scene PCD (yes, please customize the post-processing), another function that also takes nothing as input but return the robot's current joint configuration, and the robot description, i.e., URDF file.

Run the script, and follow the prompt. There're four stages: (1) collect scene PCDs. Please take as many as you can, this improves the performance. Caveat, please cover the metal part on the robot with paper, otherwise performance can be affected. (2) Manually select correspondence for initial ICP registration. (3) Gradient-based local optimization. (4) Program will repeat step 2 and 3 but warm start ICP with current best approximation of extrinsics, so no manual selection needed.

### Credit
Thanks Shaoxiong Yao for providing the idea and draft python script.

## ArUco- and ChArUco-based calibraion

No implementation yet. Ideally, the philosophy is generic and handy.
