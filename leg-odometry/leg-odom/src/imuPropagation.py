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
from utils.types import IMU, Attitude, ImuError, PVA, NormG
from utils.rot import matrix2euler



def _wrap_yaw_inplace(euler_rpy):
    euler_rpy[2] = (euler_rpy[2] + np.pi) % (2*np.pi) - np.pi




class INSMech:
    
    @staticmethod
    def insMech(pvapre: PVA, pvacur: PVA, imupre: IMU, imucur: IMU):
        imucur_dvel = imucur.acceleration * imucur.dt
        imucur_dtheta = imucur.angular_velocity * imucur.dt
        imupre_dvel = imupre.acceleration * imupre.dt
        imupre_dtheta = imupre.angular_velocity * imupre.dt

        # Calculate cross products
        temp1 = np.cross(imucur_dtheta, imucur_dvel) / 2
        temp2 = np.cross(imupre_dtheta, imucur_dvel) / 12
        temp3 = np.cross(imupre_dvel, imucur_dtheta) / 12

        d_vfb = imucur_dvel + temp1 + temp2 + temp3

        d_vfn = pvapre.att.cbn @ d_vfb

        gl = np.array([0, 0, NormG], dtype=np.float64)
        d_vgn = gl * imucur.dt

        pvacur.vel = pvapre.vel + d_vfn + d_vgn
        midvel = (pvacur.vel + pvapre.vel) / 2
        pvacur.pos = pvapre.pos + midvel * imucur.dt

        rot_bframe = imucur_dtheta + np.cross(imupre_dtheta, imucur_dtheta) / 12

        Cbb = R.from_rotvec(rot_bframe).as_matrix()
        pvacur.att.cbn = pvapre.att.cbn @ Cbb
        pvacur.att.qbn = R.from_matrix(pvacur.att.cbn)

        pvacur.att.euler = matrix2euler(pvacur.att.cbn)



    @staticmethod
    def imuCompensate(imu: IMU, imuerror: ImuError):
        imu.angular_velocity -= imuerror.gyrbias
        imu.acceleration -= imuerror.accbias

        gyrscale = np.ones(3) + imuerror.gyrscale
        accscale = np.ones(3) + imuerror.accscale

        imu.angular_velocity = imu.angular_velocity * (1 / gyrscale)
        imu.acceleration = imu.acceleration * (1 / accscale)
