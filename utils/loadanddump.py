from __future__ import annotations
import numpy as np
import sys
from typing import List
from .types import D2R, R2D, Paras,NavState


class FileSaver:
    def __init__(self, path: str, columns: int):
        self.f = open(path, "w", encoding="utf-8")
        self.columns = int(columns)
    def dump(self, data: List[float]):
        if len(data) != self.columns:
            raise ValueError(f"FileSaver columns mismatch: expect {self.columns}, got {len(data)}")
        line = "".join(f"{float(x):<15.9f} " for x in data)
        self.f.write(line.rstrip() + "\n")
    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def loadConfig(config: dict, paras: Paras) -> bool:

    try:
        initposstd_vec = list(config["initposstd"])
        initvelstd_vec = list(config["initvelstd"])
        initattstd_vec = list(config["initattstd"])
    except Exception:
        print("Failed when loading configuration. Please check initial std of position, velocity, and attitude!", file=sys.stderr)
        return False
    for i in range(3):
        paras.initstate_std.pos[i]   = float(initposstd_vec[i])
        paras.initstate_std.vel[i]   = float(initvelstd_vec[i])
        paras.initstate_std.euler[i] = float(initattstd_vec[i]) * D2R
    try:
        arw = float(config["imunoise"]["arw"])
        vrw = float(config["imunoise"]["vrw"])
        gbstd = float(config["imunoise"]["gbstd"])
        abstd = float(config["imunoise"]["abstd"])
        gsstd = float(config["imunoise"]["gsstd"])
        asstd = float(config["imunoise"]["asstd"])
        paras.imunoise.corr_time = float(config["imunoise"]["corrtime"])
    except Exception:
        print("Failed when loading configuration. Please check IMU noise!", file=sys.stderr)
        return False
    for i in range(3):
        paras.imunoise.gyr_arw[i] = arw   * (D2R / 60.0)
        paras.imunoise.acc_vrw[i] = vrw   / 60.0
        paras.imunoise.gyrbias_std[i]  = gbstd * (D2R / 3600.0)
        paras.imunoise.accbias_std[i]  = abstd * 1e-5
        paras.imunoise.gyrscale_std[i] = gsstd * 1e-6
        paras.imunoise.accscale_std[i] = asstd * 1e-6
        paras.initstate_std.imuerror.gyrbias[i]  = gbstd * (D2R / 3600.0)
        paras.initstate_std.imuerror.accbias[i]  = abstd * 1e-5
        paras.initstate_std.imuerror.gyrscale[i] = gsstd * 1e-6
        paras.initstate_std.imuerror.accscale[i] = asstd * 1e-6

    
    paras.imunoise.corr_time *= 3600.0
    paras.starttime = float(config["starttime"])
    paras.initAlignmentTime = int(config["initAlignmentTime"])
    base_in_bodyimu_vec = list(config["base_in_bodyimu"])
    paras.base_in_bodyimu = np.array(base_in_bodyimu_vec, dtype=np.float64)
    paras.robotpara.ox = float(config["robotpara"]["ox"])
    paras.robotpara.oy = float(config["robotpara"]["oy"])
    paras.robotpara.ot = float(config["robotpara"]["ot"])
    paras.robotpara.lc = float(config["robotpara"]["lc"])
    paras.robotpara.lt = float(config["robotpara"]["lt"])
    robot_b_rotmat_vec = np.array(config["rotmat"], dtype=np.float64)
    if robot_b_rotmat_vec.size != 9:
        raise ValueError("Nein!!!! rotmat must have 9 elements (必须是九个！！！).")
    paras.robotbody_rotmat = robot_b_rotmat_vec.reshape(3, 3, order="C")
    return True


def writeNavResult(time: float, navstate:NavState, navfile: FileSaver, imuerrfile: FileSaver):
    result = []
    result.append(time)
    result.append(navstate.pos[0])
    result.append(navstate.pos[1])
    result.append(navstate.pos[2])
    result.append(navstate.vel[0])
    result.append(navstate.vel[1])
    result.append(navstate.vel[2])
    result.append(navstate.euler[0] * R2D)
    result.append(navstate.euler[1] * R2D)
    result.append(navstate.euler[2] * R2D)
    navfile.dump(result)
    imuerr = navstate.imuerror
    result = []
    result.append(time)
    result.append(imuerr.gyrbias[0] * R2D * 3600.0)
    result.append(imuerr.gyrbias[1] * R2D * 3600.0)
    result.append(imuerr.gyrbias[2] * R2D * 3600.0)
    result.append(imuerr.accbias[0] * 1e5)
    result.append(imuerr.accbias[1] * 1e5)
    result.append(imuerr.accbias[2] * 1e5)
    result.append(imuerr.gyrscale[0] * 1e6)
    result.append(imuerr.gyrscale[1] * 1e6)
    result.append(imuerr.gyrscale[2] * 1e6)
    result.append(imuerr.accscale[0] * 1e6)
    result.append(imuerr.accscale[1] * 1e6)
    result.append(imuerr.accscale[2] * 1e6)
    imuerrfile.dump(result)

def writeSTD(time: float, cov: np.ndarray, stdfile: FileSaver):
    result = []
    result.append(time)
    for i in range(0, 6):
        result.append(np.sqrt(max(cov[i, i], 0.0)))
    for i in range(6, 9):
        result.append(np.sqrt(max(cov[i, i], 0.0)) * R2D)
    for i in range(9, 12):
        result.append(np.sqrt(max(cov[i, i], 0.0)) * R2D * 3600.0)
    for i in range(12, 15):
        result.append(np.sqrt(max(cov[i, i], 0.0)) * 1e5)
    for i in range(15, 21):
        result.append(np.sqrt(max(cov[i, i], 0.0)) * 1e6)
    stdfile.dump(result)
