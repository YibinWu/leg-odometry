import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_traj(traj_path: Path):
   
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)

    # t px py pz roll pitch yaw
    data = np.loadtxt(traj_path, skiprows=1)
    x, y = data[:, 1], data[:, 2]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, lw=1.0)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('XY Trajectory_python (Top View)')
    ax.grid(True)

    ax.set_aspect('equal', adjustable='box')

    dx, dy = (x.max()-x.min()), (y.max()-y.min())
    pad = 0.05 * max(dx, dy) if max(dx, dy) > 0 else 1.0
    ax.set_xlim(x.min()-pad, x.max()+pad)
    ax.set_ylim(y.min()-pad, y.max()+pad)

    plt.tight_layout()
    plt.show()
