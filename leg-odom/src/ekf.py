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
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from DATA_READING_types_1 import (
    Paras,
    IMU,
    RobotSensor,
    NavState,
    PVA,
    window_length,
    halfwindow_length,
    ImuError)
from DATA_PROCESSION_imuPropagation import INSMech
import copy
from collections import deque
from TOOLS_math_rot import euler2cbn, matrix2euler



def square(x: float | np.ndarray) -> float | np.ndarray:  # element‑wise
    return x * x

# ------------ StateID ------------在header里面
P_ID, V_ID, PHI_ID = 0, 3, 6
BG_ID, BA_ID, SG_ID, SA_ID = 9, 12, 15, 18
FL_ID, FR_ID, RL_ID, RR_ID = 21, 24, 27, 30
RANK = 33 

# ------------ NoiseID ------------
ARW_ID, VRW_ID = 3, 0
BGSTD_ID, BASTD_ID = 6, 9
SGSTD_ID, SASTD_ID = 12, 15
FL_STD_ID, FR_STD_ID, RL_STD_ID, RR_STD_ID = 18, 21, 24, 27
NOISERANK = 30

# foot block
foot_noise_block = [FL_STD_ID, FR_STD_ID, RL_STD_ID, RR_STD_ID]

def footstateid(leg: int) -> int:
    return 21 + leg*3



class EKF:
 
    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        return np.array([
            [0,     -v[2],  v[1]],
            [v[2],   0,    -v[0]],
            [-v[1],  v[0],  0   ]
        ], dtype=float)

    
    @staticmethod
    def _wrap_yaw_0_2pi(yaw_val: float) -> float:
        y = float(yaw_val) % (2.0 * np.pi)
        return y if y >= 0.0 else y + 2.0 * np.pi


    def __init__(self, paras_: Paras, dataset: np.ndarray):
        self.paras_ = paras_
        self.dataset = dataset
        
        self.imucur_, self.imupre_ = IMU(), IMU()
        self.pvacur_  = PVA()       
        self.pvapre_  = PVA()     
        self.imuerror_ = ImuError()
        self.robotsensor_=RobotSensor()

        self.foot_pos_abs_ = np.zeros((3, 4))
        self.foot_pos_rel_ = np.zeros((3,4))
        self.foot_vel_rel_ = np.zeros((3, 4))
        self.measument_updated_ = False
        self.estimated_contacts = np.zeros(4)

        self.ifinitAligned = False
        self.alignTimestamp = 0.0
        self.initAlignEpochs = 0
        self.init_gyro_mean = np.zeros(3)
        self.init_acc_mean = np.zeros(3)
        self.initfootposition_inbody = np.zeros((3, 4))

    
        self.Cov_ = np.zeros((RANK, RANK))
        self.Qc_ = np.zeros((NOISERANK, NOISERANK))
        self.delta_x_ = np.zeros((RANK, 1))

        imunoise = self.paras_.imunoise
        fac = 2.0 / imunoise.corr_time
        self.Qc_[ARW_ID:ARW_ID+3, ARW_ID:ARW_ID+3] = np.diag(square(imunoise.gyr_arw))
        self.Qc_[VRW_ID:VRW_ID+3, VRW_ID:VRW_ID+3] = np.diag(square(imunoise.acc_vrw))
        self.Qc_[BGSTD_ID:BGSTD_ID+3, BGSTD_ID:BGSTD_ID+3] = fac * np.diag(square(imunoise.gyrbias_std))
        self.Qc_[BASTD_ID:BASTD_ID+3, BASTD_ID:BASTD_ID+3] = fac * np.diag(square(imunoise.accbias_std))
        self.Qc_[SGSTD_ID:SGSTD_ID+3, SGSTD_ID:SGSTD_ID+3] = fac * np.diag(square(imunoise.gyrscale_std))
        self.Qc_[SASTD_ID:SASTD_ID+3, SASTD_ID:SASTD_ID+3] = fac * np.diag(square(imunoise.accscale_std))
        
        foot_pos_noise = 0.001
        for blk in foot_noise_block:
            self.Qc_[blk:blk + 3, blk:blk + 3] = np.eye(3) * square(foot_pos_noise)
        self.initialization(paras_.initstate,paras_.initstate_std)
 


    def initialization(self, initstate: NavState, initstate_std: NavState):
        self.pvacur_.pos = initstate.pos.copy()
        self.pvacur_.vel = initstate.vel.copy()
        self.pvacur_.att.euler = initstate.euler.copy()

        cbn0 = euler2cbn(self.pvacur_.att.euler)    # [roll,pitch,yaw]
        self.pvacur_.att.cbn = cbn0
        self.pvacur_.att.qbn = R.from_matrix(cbn0)  # 仍旧是 SciPy Rotation

        
        self.imuerror_.gyrbias  = initstate.imuerror.gyrbias.copy()
        self.imuerror_.accbias  = initstate.imuerror.accbias.copy()
        self.imuerror_.gyrscale = initstate.imuerror.gyrscale.copy()
        self.imuerror_.accscale = initstate.imuerror.accscale.copy()
        
        self.pvapre_ = copy.deepcopy(self.pvacur_)
    
        def put_diag(idx, std3):
            self.Cov_[idx:idx+3, idx:idx+3] = np.diag(np.square(std3))

        imu_std = initstate_std.imuerror
        put_diag(P_ID,   initstate_std.pos)
        put_diag(V_ID,   initstate_std.vel)
        put_diag(PHI_ID, initstate_std.euler)
        put_diag(BG_ID,  imu_std.gyrbias)
        put_diag(BA_ID,  imu_std.accbias)
        put_diag(SG_ID,  imu_std.gyrscale)
        put_diag(SA_ID,  imu_std.accscale)

        foot_pos_std = 0.01
        for idx in (FL_ID, FR_ID, RL_ID, RR_ID):
            self.Cov_[idx:idx+3, idx:idx+3] = np.eye(3) * (foot_pos_std ** 2)

    def newImuProcess(self):
        self.insPropagation(self.imupre_,self.imucur_)
        self.measUpdate()
        self.stateFeedback()
        self.checkCov()
        self.pvapre_ = copy.deepcopy(self.pvacur_)
        self.imupre_  = copy.deepcopy(self.imucur_)


    def insPropagation(self,imupre:IMU,imucur:IMU):
        INSMech.imuCompensate(imucur, self.imuerror_)
        # INSMech::insMech(pvapre_, pvacur_, imupre, imucur);对应cpp
        INSMech.insMech(self.pvapre_, self.pvacur_, imupre, imucur)
        # Eigen::MatrixXd Phi, F, Qd, G;
        Phi = np.zeros_like(self.Cov_)
        F = np.zeros_like(self.Cov_)
        Qd = np.zeros_like(self.Cov_)
        G = np.zeros((RANK, NOISERANK), dtype=np.float64)

        ff = self.robotsensor_.footforce.copy()
        th = getattr(self.paras_, "contact_force_threshold", 20.0)
        self.estimated_contacts = (ff > th).astype(float)

        F.fill(0.0); 
        Qd.fill(0.0); 
        G.fill(0.0)
        F[P_ID:P_ID+3, V_ID:V_ID+3] = np.eye(3)

        Cbn = self.pvapre_.att.cbn
        acc_b = imucur.acceleration
        F[V_ID:V_ID+3, PHI_ID:PHI_ID+3] = self.skew_symmetric(Cbn @ acc_b)
        F[V_ID:V_ID+3, BA_ID:BA_ID+3] = Cbn
        F[V_ID:V_ID+3, SA_ID:SA_ID+3] = Cbn @ np.diag(acc_b)
        F[PHI_ID:PHI_ID+3, BG_ID:BG_ID+3] = -Cbn
        omega_b = imucur.angular_velocity
        F[PHI_ID:PHI_ID+3, SG_ID:SG_ID+3] = -Cbn @ np.diag(omega_b)
        tau = self.paras_.imunoise.corr_time
        for blk in (BG_ID, BA_ID, SG_ID, SA_ID):
            F[blk:blk+3, blk:blk+3] = -np.eye(3) / tau

      
        G[V_ID:V_ID+3, VRW_ID:VRW_ID+3]   = Cbn
        G[PHI_ID:PHI_ID+3, ARW_ID:ARW_ID+3] = Cbn
        G[BG_ID:BG_ID+3,   BGSTD_ID:BGSTD_ID+3] = np.eye(3)
        G[BA_ID:BA_ID+3,   BASTD_ID:BASTD_ID+3] = np.eye(3)
        G[SG_ID:SG_ID+3,   SGSTD_ID:SGSTD_ID+3] = np.eye(3)
        G[SA_ID:SA_ID+3,   SASTD_ID:SASTD_ID+3] = np.eye(3)

        big = 1e3
        G[FL_ID:FL_ID+3, FL_STD_ID:FL_STD_ID+3] = (1 + (1 - self.estimated_contacts[0]) * big) * np.eye(3)
        G[FR_ID:FR_ID+3, FR_STD_ID:FR_STD_ID+3] = (1 + (1 - self.estimated_contacts[1]) * big) * np.eye(3)
        G[RL_ID:RL_ID+3, RL_STD_ID:RL_STD_ID+3] = (1 + (1 - self.estimated_contacts[2]) * big) * np.eye(3)
        G[RR_ID:RR_ID+3, RR_STD_ID:RR_STD_ID+3] = (1 + (1 - self.estimated_contacts[3]) * big) * np.eye(3)


        dt = imucur.dt
        Phi = np.eye(RANK) + F * dt
        Qd = G @ self.Qc_ @ G.T * dt
        Qd = 0.5 * (Phi @ Qd @ Phi.T + Qd)
        self.EKFPredict(Phi, Qd)

    def EKFPredict(self,Phi:np.ndarray, Qd:np.ndarray):
        assert Phi.shape[0] == self.Cov_.shape[0]
        assert Qd.shape[0] == self.Cov_.shape[0]

        self.Cov_ = Phi @ self.Cov_ @ Phi.T + Qd
        self.Cov_ = 0.5 * (self.Cov_ + self.Cov_.T)
        self.delta_x_ = Phi @ self.delta_x_


    def EKFUpdate(self, dz: np.ndarray, H: np.ndarray, R: np.ndarray):
        assert H.shape[1] == self.Cov_.shape[0]
        assert dz.shape[0] == H.shape[0]
        assert dz.shape[0] == R.shape[0]
        assert dz.shape[1] == 1
        S = H @ self.Cov_ @ H.T + R
        S = 0.5 * (S + S.T)
        Y = np.linalg.solve(S, H @ self.Cov_) 
        K = Y.T 
        I = np.eye(self.Cov_.shape[0])
        innov = dz - H @ self.delta_x_
        self.delta_x_ = self.delta_x_ + K @ innov
        self.Cov_ = (I - K @ H) @ self.Cov_ @ (I - K @ H).T + K @ R @ K.T
        self.Cov_ = 0.5 * (self.Cov_ + self.Cov_.T)




    def measUpdate(self):
        for i in range(4):
            if self.estimated_contacts[i] != 0:
                self.computeRelFootPosVel(self.robotsensor_, i)
                dz = np.zeros((6, 1))
                dz[0:3, 0] = (
                    self.pvacur_.att.cbn.T @ (self.foot_pos_abs_[:, i] - self.pvacur_.pos)
                    - self.foot_pos_rel_[:, i]
                )
                dz[3:6, 0] = (
                    self.pvacur_.vel
                    + self.pvacur_.att.cbn @ (
                        self.skew_symmetric(self.imucur_.angular_velocity)
                        @ self.foot_pos_rel_[:, i]
                        + self.foot_vel_rel_[:, i]
                    )
                )
                H = np.zeros((6, RANK))
                H[0:3, P_ID:P_ID+3] = -self.pvacur_.att.cbn.T
                H[0:3, PHI_ID:PHI_ID+3] = -self.pvacur_.att.cbn.T @ self.skew_symmetric(
                    self.foot_pos_abs_[:, i] - self.pvacur_.pos
                )
                H[0:3, footstateid(i):footstateid(i)+3] = self.pvacur_.att.cbn.T

                H[3:6, V_ID:V_ID+3] = np.eye(3)
                H[3:6, PHI_ID:PHI_ID+3] = self.skew_symmetric(
                    self.pvacur_.att.cbn.T @ (
                        self.skew_symmetric(self.imucur_.angular_velocity) @ self.foot_pos_rel_[:, i]
                        + self.foot_vel_rel_[:, i]
                    )
                )
                H[3:6, BG_ID:BG_ID+3] = -self.pvacur_.att.cbn.T @ self.skew_symmetric(
                    self.foot_pos_rel_[:, i]
                )
                H[3:6, SG_ID:SG_ID+3] = -self.pvacur_.att.cbn.T @ self.skew_symmetric(
                    self.foot_vel_rel_[:, i]
                ) @ np.diag(self.imucur_.angular_velocity)

                R = np.zeros((6, 6))
                R[0:3, 0:3] = np.eye(3) * (0.01 ** 2)
                R[3:6, 3:6] = np.eye(3) * (0.1 ** 2)



                self.EKFUpdate(dz, H, R)
                self.measument_updated_ = True


    def computeRelFootPosVel(self,robotsensor_: RobotSensor, legid: int):
        lfoot = 1 if legid in (0, 2) else -1
        ffoot = 1 if legid < 2 else -1
        ox, oy, ot = self.paras_.robotpara.ox, self.paras_.robotpara.oy, self.paras_.robotpara.ot
        lc, lt = self.paras_.robotpara.lc, self.paras_.robotpara.lt
        jp = robotsensor_.joint_angular_position[legid * 3 : legid * 3 + 3]
        jv = robotsensor_.joint_angular_velocity[legid * 3 : legid * 3 + 3]
        s1, s2, s3 = np.sin(jp)
        c1, c2, c3 = np.cos(jp)
        s23 = np.sin(jp[1] + jp[2])
        c23 = np.cos(jp[1] + jp[2])
        self.foot_pos_rel_[:, legid] = np.array(
            [
                -lt * s2 - lc * s23 + ffoot * ox,
                lfoot * ot * c1 + lc * s1 * c23 + lt * c2 * s1 + lfoot * oy,
                lfoot * ot * s1 - lc * c1 * c23 - lt * c1 * c2,
            ]
        )

        J = np.zeros((3, 3))
        J[0] = [0, -lc * c23 - lt * c2, -lc * c23]
        J[1] = [
            lt * c1 * c2 - lfoot * ot * s1 + lc * c1 * c23,
            -s1 * (lc * s23 + lt * s2),
            -lc * s23 * s1,
        ]
        J[2] = [
            lt * c2 * s1 + lfoot * ot * c1 + lc * s1 * c23,
            c1 * (lc * s23 + lt * s2),
            lc * s23 * c1,
        ]
        self.foot_vel_rel_[:, legid] = J @ jv

        Rb = (self.paras_.robotbody_rotmat) @ np.diag([1.0, 1.0, -1.0])
        self.foot_pos_rel_[:, legid] = Rb @ self.foot_pos_rel_[:, legid]+ self.paras_.base_in_bodyimu 
        self.foot_vel_rel_[:, legid] = Rb @ self.foot_vel_rel_[:, legid]


    def stateFeedback(self):
        if not self.measument_updated_:
            return
        self.pvacur_.pos -= self.delta_x_[P_ID:P_ID+3,0]
        self.pvacur_.vel -= self.delta_x_[V_ID:V_ID+3,0]

        delta_att = self.delta_x_[PHI_ID:PHI_ID+3,0]
        qbn = R.from_rotvec(delta_att) 
        qn_rot = self.pvacur_.att.qbn
        q_new = qbn * qn_rot
        self.pvacur_.att.qbn =  q_new
        self.pvacur_.att.cbn =  q_new.as_matrix()
        self.pvacur_.att.euler = matrix2euler(self.pvacur_.att.cbn)

        
        self.imuerror_.gyrbias += self.delta_x_[BG_ID:BG_ID+3,0]
        self.imuerror_.accbias += self.delta_x_[BA_ID:BA_ID+3,0]
        self.imuerror_.gyrscale += self.delta_x_[SG_ID:SG_ID+3,0]
        self.imuerror_.accscale += self.delta_x_[SA_ID:SA_ID+3,0]

        for leg in range(4):
            blk = footstateid(leg)
            self.foot_pos_abs_[:,leg] -= self.delta_x_[blk:blk+3,0]

        self.delta_x_[:] = 0.0
        self.measument_updated_ = False


    def getNavState(self) -> NavState:
        state = NavState()
        state.pos = self.pvacur_.pos.copy()
        state.vel = self.pvacur_.vel.copy()
        state.euler = self.pvacur_.att.euler.copy()
        state.imuerror.gyrbias = self.imuerror_.gyrbias.copy()
        state.imuerror.accbias = self.imuerror_.accbias.copy()
        state.imuerror.gyrscale = self.imuerror_.gyrscale.copy()
        state.imuerror.accscale = self.imuerror_.accscale.copy()
        return state

    def getpreNavState(self) -> NavState:
        state = NavState()
        state.pos = self.pvapre_.pos.copy()
        state.vel = self.pvapre_.vel.copy()
        state.euler = self.pvapre_.att.euler.copy()
        state.imuerror.gyrbias = self.imuerror_.gyrbias.copy()
        state.imuerror.accbias = self.imuerror_.accbias.copy()
        state.imuerror.gyrscale = self.imuerror_.gyrscale.copy()
        state.imuerror.accscale = self.imuerror_.accscale.copy()
        return state
    

#############################################################################

    def addImuData(self, imu: IMU):
        if not hasattr(self, "imuBuff_"):
            self.imuBuff_ = deque(maxlen=window_length)
        if len(self.imuBuff_) < window_length:
            self.imuBuff_.append(imu)
        else:
            self.imuBuff_.popleft()
            self.imuBuff_.append(imu)
            self.imucur_ = self.imuBuff_[halfwindow_length]
            self.imupre_ = self.imuBuff_[halfwindow_length - 1]

    def addSensorData(self, robotsensor: RobotSensor):
        self.robotsensor_ = robotsensor

    def setInitGyroBias(self, gyro_bias: np.ndarray):
        self.imuerror_.gyrbias = gyro_bias.copy()
    def setInitAttitude(self, roll: float, pitch: float):
        self.pvacur_.att.euler[:] = [roll, pitch, 0.0]
        cbn0 = euler2cbn(self.pvacur_.att.euler)
        self.pvacur_.att.cbn = cbn0
        self.pvacur_.att.qbn = R.from_matrix(cbn0)
        self.pvapre_ = copy.deepcopy(self.pvacur_)

    def setInitFootPos(self, foot_pos_3x4: np.ndarray):
        self.foot_pos_abs_ = np.asarray(foot_pos_3x4, dtype=np.float64).copy()

    def getfoot_pos_rel(self) -> np.ndarray:
        return self.foot_pos_rel_.copy()
    
    def timestamp(self) -> float:
        return float(self.imucur_.timestamp)
    
    def getCovariance(self) -> np.ndarray:
        return self.Cov_.copy()
    
    def checkCov(self):
        if np.any(np.diag(self.Cov_) < 0):
            raise ValueError(f"Covariance is negative at {self.imucur_.timestamp}!")