import sys, yaml, numpy as np
from pathlib import Path
from types import SimpleNamespace
from argparse import ArgumentParser
from tqdm import tqdm

from utils.types import Paras, IMU, RobotSensor
from utils.loadanddump import loadConfig, FileSaver, writeNavResult, writeSTD
from utils.dataloader import DataLoader
from utils.streams import make_streams_from_lowstate
from src.ekf import EKF
from src.initalign import initStaticAlignment
from utils.vis import plot_traj

def resample_to_rate(arr, target_hz, t0=None):
 
    import numpy as np
    t = arr[:, 0].astype(float)
    if t0 is None:
        t0 = np.ceil(t[0] * target_hz) / target_hz
    k = np.round((t - t0) * target_hz).astype(np.int64)
    keep_idx = np.unique(k, return_index=True)[1]
    k_keep = k[keep_idx]
    mask = k_keep >= 0
    idx = keep_idx[mask]
    arr2 = arr[idx]
    return arr2, t0

def main():
    ap = ArgumentParser()
    ap.add_argument("--config",   required=True, help="path to YAML config")
    ap.add_argument("--bagpath",  required=True, help="path to rosbag2 folder")
    ap.add_argument("--topic",    default=None, help="ROS topic (default from YAML)")
    ap.add_argument("--rate",     type=int, default=None, help="IMU rate Hz (default from YAML or 200)")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paras = Paras()
    if not loadConfig(cfg, paras):
        sys.exit(1)

    topic    = args.topic or cfg.get("topic", "/lowstate")
    imu_rate = args.rate  or int(cfg.get("imudatarate", 200))

 
    bagpath = Path(args.bagpath)
    dl = DataLoader(config=SimpleNamespace(topic=topic), data_path=bagpath)
 
    lowstate_np = dl.load_data()
    lowstate_np, t0_rs = resample_to_rate(lowstate_np, imu_rate)

   
    ts = lowstate_np[:, 0].astype(float)
    bag_start, bag_end = float(ts[0]), float(ts[-1])
    starttime = float(cfg.get("starttime", bag_start))
    endtime   = float(cfg.get("endtime",  bag_end))
    if endtime < 0: endtime = bag_end
    t0 = starttime + paras.initAlignmentTime
    print(f"[Range] bag: [{bag_start:.3f}, {bag_end:.3f}]  cfg start={starttime:.3f}  align={paras.initAlignmentTime:.1f}s  t0={t0:.3f}  end={endtime:.3f}")
    if not (bag_start <= starttime < endtime <= bag_end) or t0 >= endtime:
        print("Warning!!time stamp doesn't match, running the whole bag_time")
        starttime = bag_start
        endtime   = bag_end
        t0 = starttime + paras.initAlignmentTime

   
    imu_stream, sensor_stream = make_streams_from_lowstate(lowstate_np, imu_rate)
    ekf = EKF(paras, dataset=None)
    print("Static alignment ...")
    initStaticAlignment(ekf, imu_stream, sensor_stream, paras)

   
    mask = (lowstate_np[:, 0] >= t0) & (lowstate_np[:, 0] <= endtime)
    rows = lowstate_np[mask]
    print(f"[Range] selected {len(rows)} frames for EKF")

    
    outdir = Path(
    (cfg.get("output", {}) or {}).get("dir")
    or cfg.get("outputpath")
    or (Path.cwd() / "output")
)
    outdir.mkdir(parents=True, exist_ok=True)
    navfile    = FileSaver(str(outdir / "traj.txt"),     10)
    imuerrfile = FileSaver(str(outdir / "imuerror.txt"), 13)
    stdfile    = FileSaver(str(outdir / "std.txt"),      22)

    dt_default = 1.0 / float(imu_rate)
    for row in tqdm(rows, total=len(rows), desc="EKF", unit="frm",
                    mininterval=0.2, leave=False):
        ts = float(row[0])

      
        imu = IMU()
        imu.timestamp = ts
        imu.angular_velocity = row[1:4].astype(np.float64).copy()
        imu.acceleration = row[4:7].astype(np.float64).copy()
        imu.dt = dt_default
        
        rs = RobotSensor()
        rs.timestamp = ts
        rs.joint_angular_position = row[7:19].astype(np.float64).copy()
        rs.joint_angular_velocity = row[19:31].astype(np.float64).copy()
        rs.footforce              = row[31:35].astype(np.float64).copy()

      
        ekf.addImuData(imu)
        ekf.addSensorData(rs)
        ekf.newImuProcess()

        navstate = ekf.getNavState()
        cov  = ekf.getCovariance()
        writeNavResult(ekf.timestamp(), navstate, navfile, imuerrfile)
        writeSTD(ekf.timestamp(), cov, stdfile)

    navfile.close(); imuerrfile.close(); stdfile.close()
    print("\nWheel-INS Process Finish!")


    print("drawing...")
    CONFIG_PATH = Path(r".\config\go2.yaml")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        output_dir = Path(cfg["output"]["dir"])
    except Exception as e:
        raise KeyError(f"YAML is not right：{e}")

    traj_file = output_dir / "traj.txt"
    print(f"[INFO] Using traj file: {traj_file}")

    plot_traj(traj_file)



if __name__ == "__main__":
    main()
