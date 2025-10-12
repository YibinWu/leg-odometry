# math_rot.py
import numpy as np

def euler2cbn(euler_rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = float(euler_rpy[0]), float(euler_rpy[1]), float(euler_rpy[2])
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float64)
    Ry = np.array([[ cp, 0, sp],
                   [ 0, 1, 0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,  0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx

def matrix2euler(cbn: np.ndarray) -> np.ndarray:
    roll  = np.arctan2(cbn[2,1], cbn[2,2])
    pitch = -np.arcsin(cbn[2,0])
    yaw   = np.arctan2(cbn[1,0], cbn[0,0])  # (-pi, pi]
    return np.array([roll, pitch, yaw], dtype=np.float64)

