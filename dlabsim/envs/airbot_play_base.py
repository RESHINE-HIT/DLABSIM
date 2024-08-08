import mujoco
import numpy as np

from dlabsim.utils.base_config import BaseConfig
from dlabsim.envs.simulator import SimulatorBase

class AirbotPlayCfg(BaseConfig):
    expreriment    = "qiuzhi_11F_il"
    robot          = "airbot_play"
    mjcf_file_path = "mjcf/il.xml"
    decimation     = 4
    timestep       = 0.005
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 200,
        "width"  : 640, # 640, #1920
        "height" : 480, # 480, #1080
    }
    put_text       = False
    obs_camera_id  = 1
    rb_link_list   = []
    obj_list       = []
    init_joint_pose = {
        "joint1"  :  0.06382703,
        "joint2"  : -0.71966516,
        "joint3"  :  1.2772779,
        "joint4"  : -1.5965166,
        "joint5"  :  1.72517278,
        "joint6"  :  1.80462028,
        "gripper" :  0.455,
    }

class AirbotPlayBase(SimulatorBase):
    def __init__(self, config: AirbotPlayCfg):
        self.nj = 7
        super().__init__(config)

        self.jq = np.zeros(self.nj)
        self.jv = np.zeros(self.nj)

        self.init_joint_pose = []
        for i in range(self.nj-1):
            self.init_joint_pose.append(self.config.init_joint_pose["joint{}".format(i+1)])
        self.init_joint_pose.append(self.config.init_joint_pose["gripper"] * 0.04)
        self.init_joint_pose = np.array(self.init_joint_pose)

        self.resetState()
        self.updateState()


    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.teleop:
            self.teleop.reset()

        self.jq = np.zeros(self.nj)
        self.jv = np.zeros(self.nj)

        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.nj] = self.init_joint_pose.copy()

        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateState(self):
        self.jq = self.mj_data.qpos[:self.nj]
        self.jv = self.mj_data.qvel[:self.nj]

    def updateControl(self, action):
        if self.mj_data.qpos[self.nj-1] < 0.0:
            self.mj_data.qpos[self.nj-1] = 0.0

        for i in range(self.nj):
            if i == self.nj-1:
                self.mj_data.ctrl[i] = action[i] * 0.04 # gripper action ionverse normalization
            elif i in {3,5}:
                self.mj_data.ctrl[i] = self.step_func(self.mj_data.ctrl[i], action[i], 16 * self.mj_model.opt.timestep)
            else:
                self.mj_data.ctrl[i] = action[i]
            self.mj_data.ctrl[i] = np.clip(self.mj_data.ctrl[i], self.mj_model.actuator_ctrlrange[i][0], self.mj_model.actuator_ctrlrange[i][1])

    def step_func(self, current, target, step):
        if current < target - step:
            return current + step
        elif current > target + step:
            return current - step
        else:
            return target

    def checkTerminated(self):
        return False

    def post_physics_step(self):
        pass

    def getObservation(self):
        self.obs = {
            "jq"  : self.jq.tolist(),
            "jv"  : self.jv.tolist(),
            "img" : self.img_rgb_obs
        }
        self.obs["jq"][6] *= 25.0 # gripper normalization
        self.obs["jv"][6] *= 25.0 # gripper normalization
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

if __name__ == "__main__":
    exec_node = AirbotPlayBase(AirbotPlayCfg())

    obs = exec_node.reset()
    print(obs.keys())

    action = exec_node.init_joint_pose[:exec_node.nj]
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)
