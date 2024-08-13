import shutil
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import json
import mediapy

import multiprocessing as mp

from dlabsim.airbot_play import AirbotPlayFIK
from dlabsim import DLABSIM_ROOT_DIR, DLABSIM_ASSERT_DIR
from dlabsim.envs.airbot_play_base import AirbotPlayBase, AirbotPlayCfg

data_set_size = 200

class SimNode(AirbotPlayBase):
    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.teleop:
            self.teleop.reset()

        self.jq = np.zeros(self.nj)
        self.jv = np.zeros(self.nj)
        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.nj] = self.init_joint_pose.copy()
        
        self.mj_data.qpos[self.nj+1] = 0.2 + (np.random.random() - 1) * 0.1 - 0.06
        self.mj_data.qpos[self.nj+2] = 0.1 + (np.random.random() - 0.5) * 0.1

        self.mj_data.qpos[self.nj+8] = 0.2 + (np.random.random() - 0) * 0.1 + 0.06
        self.mj_data.qpos[self.nj+9] = 0.1 + (np.random.random() - 0.5) * 0.1

        mujoco.mj_forward(self.mj_model, self.mj_data)

    def getObservation(self):
        obj_pose = {}
        for name in self.config.obj_list:
            obj_pose[name] = self.getObjPose(name)

        self.obs = {
            "time"     : self.mj_data.time,
            "jq"       : self.jq.tolist(),
            "img"      : self.img_rgb_obs,
            "obj_pose" : obj_pose
        }
        self.obs["jq"][6] *= 25.0 # gripper normalization
        return self.obs

def recoder(save_path, obs_lst, act_lst):
    os.mkdir(save_path)
    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        obj = {
            "time" : [o['time'] for o in obs_lst],
            "obs"  : {
                "jq" : [o['jq'] for o in obs_lst],
            },
            "act"  : act_lst
        }
        json.dump(obj, fp)

    mediapy.write_video(os.path.join(save_path, "video.mp4"), [o['img'] for o in obs_lst], fps=cfg.render_set["fps"])


if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = AirbotPlayCfg()
    cfg.expreriment  = "act_airbot_play"
    cfg.rb_link_list = []
    cfg.obj_list     = ["cup_blue", "cup_pink"]
    cfg.sync         = False
    cfg.headless     = False
    cfg.decimation   = 4
    cfg.render_set   = {
        "fps"    : 50,
        "width"  : 640,
        "height" : 480
        # "width"  : 1920,
        # "height" : 1080
    }
    cfg.obs_camera_id   = 1
    cfg.init_joint_pose = {
        "joint1"  :  0.06382703,
        "joint2"  : -0.71966516,
        "joint3"  :  1.2772779,
        "joint4"  : -1.5965166,
        "joint5"  :  1.72517278,
        "joint6"  :  1.80462028,
        "gripper" :  0.5
    }

    sim_node = SimNode(cfg)

    base_posi, base_quat = sim_node.getObjPose("arm_base")
    Tmat_base = np.eye(4)
    Tmat_base[:3,:3] = Rotation.from_quat(base_quat[[1,2,3,0]]).as_matrix()
    Tmat_base[:3, 3] = base_posi
    Tmat_base_inv = np.linalg.inv(Tmat_base)

    arm_rot_mat = np.array([
        [ 0., -0.,  1.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  0.]
    ])

    tar_end_rot = np.array([
        [ 0., -0.,  1.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  0.]
    ])

    pick_pose_bias_above = [0.0, 0.04, 0.15]
    Tmat_move_bias = np.eye(4)
    Tmat_move_bias[:3,3] = pick_pose_bias_above

    state_idx = 0

    urdf_path = os.path.join(DLABSIM_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
    arm_fik = AirbotPlayFIK(urdf_path)

    obs = sim_node.reset()

    data_idx = 0
    obs_lst, act_lst = [], []

    save_dir = os.path.join(DLABSIM_ROOT_DIR, f"data/{cfg.expreriment}")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    process_list = []

    action = sim_node.init_joint_pose[:sim_node.nj].copy()
    action[6] *= 25.0

    init_control = action.copy()
    tarjq = init_control.copy()

    while sim_node.running:
        try:
            if state_idx == 0: # move to cup_blue above
                posi, quat = obs["obj_pose"]["cup_blue"]
                Tmat_cup_b_global = np.eye(4)
                Tmat_cup_b_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_cup_b_global[:3, 3] = posi
                Tmat_cup_b_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_cup_b_global

                tar_end_pose = Tmat_cup_b_local[:3, 3]
                rot = tar_end_rot @ arm_rot_mat 

                tarjq[:6] = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])

                state_idx += 1

            elif state_idx == 1: # moving to cup_blue above
                if np.allclose(obs["jq"][:6], tarjq[:6], atol=1e-2):
                    state_idx += 1

            elif state_idx == 2: # move to cup_blue
                posi, quat = obs["obj_pose"]["cup_blue"]
                Tmat_cup_b_global = np.eye(4)
                Tmat_cup_b_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_cup_b_global[:3, 3] = posi
                Tmat_move_bias[2,3] = 0.045

                Tmat_cup_b_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_cup_b_global
                tar_end_pose = Tmat_cup_b_local[:3, 3]
                rot = tar_end_rot @ arm_rot_mat 

                tarjq[:6] = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])
                state_idx += 1

            elif state_idx == 3: # move to cup_blue
                if np.allclose(obs["jq"][:6], tarjq[:6], atol=1e-2):
                    state_idx += 1

            elif state_idx == 4: # close gripper
                tarjq[6] = 0.0
                state_cnt = 0
                state_idx += 1

            elif state_idx == 5: # closing gripper
                state_cnt += 1
                if state_cnt * sim_node.config.decimation * sim_node.mj_model.opt.timestep > 0.5:
                    state_idx += 1

            elif state_idx == 6: # pickup cup_blue
                posi, quat = obs["obj_pose"]["cup_blue"]
                Tmat_cup_b_global = np.eye(4)
                Tmat_cup_b_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_cup_b_global[:3, 3] = posi
                Tmat_move_bias[2,3] = 0.18

                Tmat_cup_b_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_cup_b_global

                tar_end_pose = Tmat_cup_b_local[:3, 3]
                rot = tar_end_rot @ arm_rot_mat 

                tarjq[:6] = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])

                state_idx += 1

            elif state_idx == 7: # picking up cup_blue
                if np.allclose(obs["jq"][:6], tarjq[:6], atol=1e-2):
                    state_idx += 1

            elif state_idx == 8: # move the cup_blue to cup_pink
                posi, quat = obs["obj_pose"]["cup_pink"]
                Tmat_cup_b_global = np.eye(4)
                Tmat_cup_b_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_cup_b_global[:3, 3] = posi
                Tmat_move_bias[2,3] = 0.18
                Tmat_cup_b_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_cup_b_global

                tar_end_pose = Tmat_cup_b_local[:3, 3]
                rot = tar_end_rot @ arm_rot_mat 

                tarjq[:6] = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])

                state_idx += 1

            elif state_idx == 9: # open gripper
                if np.allclose(obs["jq"][:6], tarjq[:6], atol=1e-2):
                    tarjq[6] = 0.5
                    state_cnt = 0
                    state_idx += 1

            elif state_idx == 10: # opening gripper
                state_cnt += 1
                if state_cnt * sim_node.config.decimation * sim_node.mj_model.opt.timestep > 0.25:
                    state_idx += 1

        except ValueError:
            state_cnt = 0
            tarjq = init_control.copy()
            action = init_control.copy()

            state_idx = 0
            sim_node.reset()
            obs_lst, act_lst = [], []

        for i in range(6):
            action[i] = sim_node.step_func(action[i], tarjq[i], 2.0 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
        action[6] = sim_node.step_func(action[6], tarjq[6], 5.0 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)

        obs_lst.append(obs)
        act_lst.append(action.tolist())

        if state_idx > 10:
            save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
            process = mp.Process(target=recoder, args=(save_path, obs_lst, act_lst))
            process.start()
            process_list.append(process)

            data_idx += 1
            print("\r{:4}/{:4}".format(data_idx, data_set_size), end="")

            if data_idx >= data_set_size:
                break


            state_idx = 0
            sim_node.reset()

            obs_lst, act_lst= [], []

            tarjq = init_control.copy()
            action = init_control.copy()

        obs, pri_obs, rew, ter, info = sim_node.step(action)

    print("")
    for p in process_list:
        p.join()
