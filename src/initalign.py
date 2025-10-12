import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Iterable
from utils.types import IMU, RobotSensor, Paras
from .ekf import EKF
from tqdm import tqdm
from utils.rot import euler2cbn


def initStaticAlignment(
    ekf: EKF,
    imu_stream: Iterable[IMU],
    sensor_stream: Iterable[RobotSensor],
    paras: Paras,
):
    imu_cur = next(imu_stream)
    while imu_cur.timestamp < paras.starttime:
        imu_cur = next(imu_stream)
    sensor_cur = next(sensor_stream)
    while sensor_cur.timestamp < paras.starttime:
        sensor_cur = next(sensor_stream)

    t_end = paras.starttime + paras.initAlignmentTime

    k = 0
    gyro_sum = np.zeros(3)
    acc_sum  = np.zeros(3)
    while ekf.timestamp() < t_end:
        ekf.addImuData(imu_cur)
        gyro_sum += imu_cur.angular_velocity
        acc_sum  += imu_cur.acceleration
        k += 1
        imu_cur = next(imu_stream)

    gyro_mean = gyro_sum / max(k, 1)
    acc_mean  = acc_sum  / max(k, 1)
    roll  = np.arctan2(-acc_mean[1], -acc_mean[2])
    pitch = np.arctan2( acc_mean[0], np.sqrt(acc_mean[1]**2 + acc_mean[2]**2))

    j = 0
    footpos_in_body = np.zeros((3, 4))
    while sensor_cur.timestamp < t_end:
        ekf.addSensorData(sensor_cur)
        for leg in range(4):
            ekf.computeRelFootPosVel(sensor_cur, leg) 
        footpos_in_body += ekf.getfoot_pos_rel()
        j += 1
        sensor_cur = next(sensor_stream)

    footpos_in_body /= max(j, 1)
    Cbn0 = euler2cbn(np.array([roll, pitch, 0.0], dtype=np.float64))
    footpos_in_body = Cbn0 @ footpos_in_body
    
    ekf.setInitFootPos(footpos_in_body)
    ekf.setInitGyroBias(gyro_mean)
    ekf.setInitAttitude(roll, pitch)
 