######ImuFileLoader 和 RobotSensorLoader
# streams.py
import numpy as np
from typing import Iterable, Iterator, Tuple
from DATA_READING_types_1 import IMU, RobotSensor


def make_streams_from_lowstate(lowstate_np: np.ndarray, imu_rate_hz: float):
    assert lowstate_np.shape[1] >= 35
    dt_default = 1.0 / float(imu_rate_hz)

   
    def imu_gen():
        for row in lowstate_np:
            imu = IMU()
            ts = float(row[0])
            imu.timestamp = ts
            imu.angular_velocity = row[1:4].astype(np.float64).copy()
            imu.acceleration = row[4:7].astype(np.float64).copy()
            imu.dt = dt_default
            
            yield imu

    def sensor_gen():
        for row in lowstate_np:
            rs = RobotSensor()
            rs.timestamp = float(row[0])
            rs.joint_angular_position = row[7:19].astype(np.float64).copy()
            rs.joint_angular_velocity = row[19:31].astype(np.float64).copy()
            rs.footforce = row[31:35].astype(np.float64).copy()
            yield rs

    return imu_gen(), sensor_gen()

