# leg-odometry
<p align="left">
<a href="https://arxiv.org/pdf/2503.04580"><img src="https://img.shields.io/badge/arXiv-0078D6" /></a>
<a href="https://github.com/YibinWu/leg-odometry"><img src="https://img.shields.io/badge/python-DD78D6" /></a>
<a href="https://docs.ros.org/en/foxy/index.html"><img src="https://img.shields.io/badge/ROS2-FCC624" /></a>
</p>

## EKF-based proprioceptive state estimation for legged robots using IMU and joint encoders
This is our implementation to estimate the state of the legged robot's main body with a body mounted IMU and the joint encoders [[1](#2-reference), [2](#2-reference)]. We use the rosbag (ros2) collected from an [unitree go2](https://github.com/unitreerobotics/unitree_ros2) robot. We don't rely on a ROS2 environment since we decode the rosbag into numpy data with python.

## 💥News💥
*Jun. 2025* :tada::tada: Our [DogLegs](https://arxiv.org/pdf/2503.04580) paper that adds multiple leg-mounted IMUs to this work has been accepted to IROS 2025. Please condsider cite our paper if you find this project helpful for your research.

```bibtex
@inproceedings{wu2025iros,
  author={Wu, Yibin and Kuang, Jian and Khorshidi, Shahram and Niu, Xiaoji and Klingbeil, Lasse and Bennewitz, Maren and Kuhlmann, Heiner},
  booktitle={IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)}, 
  title={{DogLegs}: Robust Proprioceptive State Estimation for Legged Robots Using Multiple Leg-Mounted IMUs}, 
  year={2025}
}
```

### 0. Download the test data

Download the unitree go2 rosbag with [this link](https://github.com/YibinWu/leg-odometry/releases/tag/test_v1.0).


### 1. Run the code:

Make sure you have the right python enviroment before running the code.

```
python -u main.py --config "your\path\to\go2.yaml" --bagpath "your\path\to\rosbag2_2025_02_19-22_46_51 (here is the given test file)" --topic /lowstate
```

The estimated trajectory for test file should be:

<img width="600" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/5e6b3e12-aabd-4f6f-8c71-4b553476a118" />


### 2. Reference
[1] M. Bloesch, M. Hutter, M. A. Hoepflinger, S. Leutenegger, C. Gehring, C. D. Remy, and R. Siegwart, “State estimation for legged robots: Consistent fusion of leg kinematics and IMU,” in Proc. Robot.: Sci. Syst., pp. 17–24, 2013.

[2] M. Bloesch, C. Gehring, P. Fankhauser, M. Hutter, M. A. Hoepflinger, and R. Siegwart, “State estimation for legged robots on unstable and slippery terrain,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., pp. 6058–6064, 2013.