import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.io import loadmat


data = loadmat(os.path.join(os.path.dirname(__file__), 'sim_data.mat'))
t = data['tSol']
theta = data['thetaSol']
x = data['xSol']

model = mujoco.MjModel.from_xml_path('slider_crank.xml')
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)
time_ratio = 1

# while viewer.is_running():
_t_start = time.time()
for i, _t in enumerate(t):
    if time.time() - _t_start > _t / time_ratio:
        print("Running slower than data!")
    while time.time() - _t_start < _t / time_ratio:
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)

    data.ctrl[:] = -theta[i]
    print(f"t: {_t}, theta: {theta[i]}", end="\r")
