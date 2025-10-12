# MIT License
#
# Copyright (c) 2025 Yibin Wu.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
 
import numpy as np
from scipy.spatial.transform import Rotation as R
DT = np.float64
D2R = np.pi / 180.0
R2D = 180.0 / np.pi
IMU_RATE = 200
window_length = 11
halfwindow_length = window_length // 2
NormG = 9.782940329221166

class IMU:
    def __init__(self):
        self.timestamp = 0.0
        self.dt = 0.0
        self.angular_velocity = np.zeros(3, dtype=DT)  # Eigen::Vector3d
        self.acceleration = np.zeros(3, dtype=DT)  # Eigen::Vector3d


class Attitude:
    def __init__(self):
        self.qbn = R.from_quat([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        self.cbn = np.eye(3, dtype=DT)  # Identity matrix
        self.euler = np.zeros(3, dtype=DT)  # Euler angles


class PVA:
    def __init__(self):
        self.pos = np.zeros(3,dtype=DT)
        self.vel = np.zeros(3,dtype=DT)
        self.att = Attitude()



class RobotSensor:
    def __init__(self):
        self.timestamp = 0.0
        self.joint_angular_position = np.zeros(12,dtype=DT)  # Eigen::Vector12d
        self.joint_angular_velocity = np.zeros(12,dtype=DT)  # Eigen::Vector12d
        self.footforce = np.zeros(4,dtype=DT)  # Eigen::Vector4d


class ImuError:
    def __init__(self):
        self.gyrbias = np.zeros(3,dtype=DT)  # Eigen::Vector3d
        self.accbias = np.zeros(3,dtype=DT)  # Eigen::Vector3d
        self.gyrscale = np.zeros(3,dtype=DT)  # Eigen::Vector3d
        self.accscale = np.zeros(3,dtype=DT)  # Eigen::Vector3d


class NavState:
    def __init__(self):
        self.pos = np.zeros(3,dtype=DT)  # Eigen::Vector3d
        self.vel = np.zeros(3,dtype=DT)  # Eigen::Vector3d
        self.euler = np.zeros(3,dtype=DT)  # Eigen::Vector3d
        self.imuerror = ImuError()


class ImuNoise:
    def __init__(self):
        self.gyr_arw = np.zeros(3,dtype=DT)  # Eigen::Vector3d
        self.acc_vrw = np.zeros(3,dtype=DT)  # Eigen::Vector3d
        self.gyrbias_std = np.zeros(3,dtype=DT)  # Eigen::Vector3d
        self.accbias_std = np.zeros(3,dtype=DT)  # Eigen::Vector3d
        self.gyrscale_std = np.zeros(3,dtype=DT)  # Eigen::Vector3d
        self.accscale_std = np.zeros(3,dtype=DT)  # Eigen::Vector3d
        self.corr_time = 0.0


class RobotPara:
    def __init__(self):
        self.ox = 0.0
        self.oy = 0.0
        self.ot = 0.0
        self.lc = 0.0
        self.lt = 0.0

class Paras:
    def __init__(self):
        # initial state and state standard deviation
        self.initstate = NavState()
        self.initstate_std = NavState()

        # imu noise parameters
        self.imunoise = ImuNoise()

        self.starttime = 0.0

        self.initAlignmentTime = 0

        self.robotbody_rotmat = np.eye(3, dtype=DT)

        self.base_in_bodyimu = np.zeros(3,dtype=DT)

        self.robotpara = RobotPara()
       