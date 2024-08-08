import shutil
import mujoco
import numpy as np
from enum import Enum, auto
from scipy.spatial.transform import Rotation

import os
import json
import mediapy
import traceback

from dlabsim.airbot_play import AirbotPlayFIK
from dlabsim import DLABSIM_ROOT_DIR, DLABSIM_ASSERT_DIR
from dlabsim.envs.mmk2_base import MMK2Base, MMK2Cfg

class SimNode(MMK2Base):
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
        return self.obs

    def updateControl(self, action):
        super().updateControl(action)
        # self.mj_data.qpos[:7] = self.init_joint_pose[:7].copy()
        # self.mj_data.qvel[:6] = 0.0

# ValueError: Fail to solve inverse kinematics: pos=[0.521 0.037 0.367], ori=[[-0.707 -0.707  0.   ]
#  [ 0.     0.     1.   ]
#  [-0.707  0.707  0.   ]]


class StateBuildBlocks(Enum):
    SBB_SLEEPING                = auto()
    SBB_LIFT_DOWN               = auto()
    SBB_LIFT_DOWN_ING           = auto()
    SBB_MOVE_TO_CUBE_ABOVE      = auto()
    SBB_MOVE_TO_CUBE_ABOVE_ING  = auto()
    SBB_MOVE_TO_CUBE            = auto()
    SBB_MOVE_TO_CUBE_ING        = auto()
    SBB_CLOSE_GRIPPER           = auto()
    SBB_CLOSE_GRIPPER_ING       = auto()
    SBB_PICK_UP                 = auto()
    SBB_PICK_UP_ING             = auto()
    SBB_LIFT_UP                 = auto()
    SBB_LIFT_UP_ING             = auto()
    SBB_MOVE_TO_TARGET          = auto()
    SBB_MOVE_TO_TARGET_ING      = auto()
    SBB_OPEN_GRIPPER            = auto()
    SBB_OPEN_GRIPPER_ING        = auto()
    SBB_ARM_BACK                = auto()
    SBB_ARM_BACK_ING            = auto()
    SBB_END = auto()

if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = MMK2Cfg()
    cfg.expreriment  = "act_mmk2"
    cfg.mjcf_file_path = "mjcf/mmk2_blocks.xml"
    cfg.rb_link_list = []
    cfg.obj_list     = [
        "B0_0_1", "B1_0_1", "B2_0_1", "B3_0_1",
        "B0_1_1", "B1_1_1", "B2_1_1", "B3_1_1",
        "B0_2_1", "B1_2_1", "B2_2_1", "B3_2_1",
        "B0_3_1", "B1_3_1", "B2_3_1", "B3_3_1",
        "B0_4_1", "B1_4_1", "B2_4_1", "B3_4_1",
        "B0_5_1", "B1_5_1", "B2_5_1", "B3_5_1",
    ]

    cfg.sync         = False
    cfg.headless     = False
    cfg.decimation   = 4
    cfg.put_text       = True
    cfg.render_set   = {
        "fps"    : 5,
        "width"  : 1920,
        "height" : 1080,
        # "width"  : 1280,
        # "height" : 720,
    }
    cfg.obs_camera_id   = 1

    sim_node = SimNode(cfg)

    # sim_node.options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    # sim_node.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # sim_node.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # # options.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    # # options.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    # # options.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
    # sim_node.options.frame = mujoco.mjtFrame.mjFRAME_BODY.value

    cube_idx = 0

    arm_rot_mat = np.array([
        [ 0., -0.,  1.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  0.]
    ])

    tar_end_rot_left = np.array([
        [ 0., -0.707,  0.707],
        [ 1.,  0.   ,  0.   ],
        [ 0.,  0.707,  0.707],
    ])

    tar_end_rot_right = np.array([
        [ 0.,  0.707,  0.707],
        [-1.,  0.   ,  0.   ],
        [ 0., -0.707,  0.707],
    ])

    Tmat_move_bias = np.eye(4)

    state_sbb = StateBuildBlocks.SBB_SLEEPING

    urdf_path = os.path.join(DLABSIM_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
    arm_fik = AirbotPlayFIK(urdf_path)

    obs = sim_node.reset()


    data_idx = 0
    data_set_size = 1
    obs_lst, act_lst = [], []

    save_dir = os.path.join(DLABSIM_ROOT_DIR, f"data/{cfg.expreriment}")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    action = sim_node.init_ctrl.copy()
    chassis_cmd = action[:2]
    lift_cmd = action[2:3]
    head_cmd = action[3:5]
    left_arm_cmd = action[5:12]
    right_arm_cmd = action[12:19]

    chassis_cmd_buf = chassis_cmd.copy()
    lift_cmd_buf = lift_cmd.copy()
    head_cmd_buf = head_cmd.copy()
    left_arm_cmd_buf = left_arm_cmd.copy()
    right_arm_cmd_buf = right_arm_cmd.copy()

    head_cmd_buf[1] = 0.55

    input("Press Enter to continue...")

    while sim_node.running:
        if state_sbb == StateBuildBlocks.SBB_SLEEPING: # move to cube above
            state_sbb = StateBuildBlocks(state_sbb.value+1)

        try:
            if state_sbb == StateBuildBlocks.SBB_LIFT_DOWN: # move to cube above
                lift_cmd_buf[0] = 0.75
                left_arm_cmd_buf[6] = 0.035
                right_arm_cmd_buf[6] = 0.035
                state_sbb = StateBuildBlocks(state_sbb.value+1)
                print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_LIFT_DOWN_ING:
                if np.allclose(obs["jq"][7+2], lift_cmd_buf[0], atol=1e-2):
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_MOVE_TO_CUBE_ABOVE:
                pick_obj_name = cfg.obj_list[cube_idx]

                posi, quat = obs["obj_pose"][pick_obj_name]
                Tmat_cup_b_global = np.eye(4)
                Tmat_cup_b_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_cup_b_global[:3, 3] = posi

                arm_site_name = "lft_arm_base_link" if posi[1] > -0.0375 else "rgt_arm_base_link"

                base_posi, base_quat = sim_node.getObjPose(arm_site_name)
                Tmat_base = np.eye(4)
                Tmat_base[:3,:3] = Rotation.from_quat(base_quat[[1,2,3,0]]).as_matrix()
                Tmat_base[:3, 3] = base_posi
                Tmat_base_inv = np.linalg.inv(Tmat_base)

                Tmat_move_bias[1,3] = 0.1
                if "lft" in arm_site_name:
                    Tmat_move_bias[1,3] *= -1

                Tmat_cup_b_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_cup_b_global

                tar_end_pose = Tmat_cup_b_local[:3, 3]
                if "lft" in arm_site_name:
                    rot = tar_end_rot_left @ arm_rot_mat
                else:
                    rot = tar_end_rot_right @ arm_rot_mat

                jres = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])

                if "lft" in arm_site_name:
                    left_arm_cmd_buf[:6] = jres
                    right_arm_cmd_buf[:6] = [0.0, -0.847,  1.2 ,  0.0, -1.5708, -0.88] 
                else:
                    left_arm_cmd_buf[:6] = [0.0, -0.847,  1.2 ,  0.0, 1.5708, 0.88]
                    right_arm_cmd_buf[:6] = jres

                state_sbb = StateBuildBlocks(state_sbb.value+1)

            elif state_sbb == StateBuildBlocks.SBB_MOVE_TO_CUBE_ABOVE_ING:
                if np.allclose(obs["jq"][12:18], left_arm_cmd_buf[:6], atol=1e-2) and np.allclose(obs["jq"][20:26], right_arm_cmd_buf[:6], atol=1e-2):
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_MOVE_TO_CUBE:
                posi, quat = obs["obj_pose"][pick_obj_name]
                Tmat_cup_b_global = np.eye(4)
                Tmat_cup_b_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_cup_b_global[:3, 3] = posi

                base_posi, base_quat = sim_node.getObjPose(arm_site_name)
                Tmat_base = np.eye(4)
                Tmat_base[:3,:3] = Rotation.from_quat(base_quat[[1,2,3,0]]).as_matrix()
                Tmat_base[:3, 3] = base_posi
                Tmat_base_inv = np.linalg.inv(Tmat_base)

                Tmat_move_bias[1,3] = 0.02
                if "lft" in arm_site_name:
                    Tmat_move_bias[1,3] *= -1

                Tmat_cup_b_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_cup_b_global
                tar_end_pose = Tmat_cup_b_local[:3, 3]
                if "lft" in arm_site_name:
                    rot = tar_end_rot_left @ arm_rot_mat
                else:
                    rot = tar_end_rot_right @ arm_rot_mat

                jres = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])
                if "lft" in arm_site_name:
                    left_arm_cmd_buf[:6] = jres
                else:
                    right_arm_cmd_buf[:6] = jres

                state_sbb = StateBuildBlocks(state_sbb.value+1)
                print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_MOVE_TO_CUBE_ING:
                if np.allclose(obs["jq"][12:18], left_arm_cmd_buf[:6], atol=1e-2) and np.allclose(obs["jq"][20:26], right_arm_cmd_buf[:6], atol=1e-2):
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_CLOSE_GRIPPER:
                if "lft" in arm_site_name:
                    left_arm_cmd_buf[6] = 0.0
                else:
                    right_arm_cmd_buf[6] = 0.0
                state_cnt = 0
                state_sbb = StateBuildBlocks(state_sbb.value+1)
                print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_CLOSE_GRIPPER_ING:
                state_cnt += 1
                if state_cnt * sim_node.config.decimation * sim_node.mj_model.opt.timestep > 0.5:
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_PICK_UP:
                posi, quat = obs["obj_pose"][pick_obj_name]
                Tmat_cup_b_global = np.eye(4)
                Tmat_cup_b_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_cup_b_global[:3, 3] = posi

                base_posi, base_quat = sim_node.getObjPose(arm_site_name)
                Tmat_base = np.eye(4)
                Tmat_base[:3,:3] = Rotation.from_quat(base_quat[[1,2,3,0]]).as_matrix()
                Tmat_base[:3, 3] = base_posi
                Tmat_base_inv = np.linalg.inv(Tmat_base)

                Tmat_move_bias[1,3] = 0.1
                if "lft" in arm_site_name:
                    Tmat_move_bias[1,3] *= -1

                Tmat_cup_b_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_cup_b_global

                tar_end_pose = Tmat_cup_b_local[:3, 3]
                if "lft" in arm_site_name:
                    rot = tar_end_rot_left @ arm_rot_mat
                else:
                    rot = tar_end_rot_right @ arm_rot_mat

                jres = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])
                if "lft" in arm_site_name:
                    left_arm_cmd_buf[:6] = jres
                else:
                    right_arm_cmd_buf[:6] = jres

                state_sbb = StateBuildBlocks(state_sbb.value+1)
                print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_PICK_UP_ING:
                if np.allclose(obs["jq"][12:18], left_arm_cmd_buf[:6], atol=1e-2) and np.allclose(obs["jq"][20:26], right_arm_cmd_buf[:6], atol=1e-2):
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_LIFT_UP:
                lift_cmd_buf[0] = 0.75 - 0.025 * cube_idx
                state_sbb = StateBuildBlocks(state_sbb.value+1)

            elif state_sbb == StateBuildBlocks.SBB_LIFT_UP_ING:
                if np.allclose(obs["jq"][7+2], lift_cmd_buf[0], atol=1e-2):
                    state_sbb = StateBuildBlocks(state_sbb.value+1)

            elif state_sbb == StateBuildBlocks.SBB_MOVE_TO_TARGET: # move the cup_blue to cup_pink
                posi = np.array([0.6, -0.01,  0.322 + 0.025 * cube_idx])
                quat = np.array([1.0, 0.0, 0.0, 0.0])
                Tmat_cup_b_global = np.eye(4)
                Tmat_cup_b_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_cup_b_global[:3, 3] = posi

                base_posi, base_quat = sim_node.getObjPose(arm_site_name)
                Tmat_base = np.eye(4)
                Tmat_base[:3,:3] = Rotation.from_quat(base_quat[[1,2,3,0]]).as_matrix()
                Tmat_base[:3, 3] = base_posi
                Tmat_base_inv = np.linalg.inv(Tmat_base)

                Tmat_move_bias[1,3] = 0.05
                if "lft" in arm_site_name:
                    Tmat_move_bias[1,3] *= -1

                Tmat_cup_b_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_cup_b_global

                tar_end_pose = Tmat_cup_b_local[:3, 3]
                if "lft" in arm_site_name:
                    rot = tar_end_rot_left @ arm_rot_mat
                else:
                    rot = tar_end_rot_right @ arm_rot_mat

                jres = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])
                if "lft" in arm_site_name:
                    left_arm_cmd_buf[:6] = jres
                else:
                    right_arm_cmd_buf[:6] = jres

                state_sbb = StateBuildBlocks(state_sbb.value+1)
                print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_MOVE_TO_TARGET_ING:
                if np.allclose(obs["jq"][12:18], left_arm_cmd_buf[:6], atol=1e-2) and np.allclose(obs["jq"][20:26], right_arm_cmd_buf[:6], atol=1e-2):
                    state_cnt = 0
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_OPEN_GRIPPER: # opening gripper
                state_cnt += 1
                if state_cnt * sim_node.config.decimation * sim_node.mj_model.opt.timestep > 0.25:
                    if "lft" in arm_site_name:
                        left_arm_cmd_buf[6] = 0.035
                    else:
                        right_arm_cmd_buf[6] = 0.035
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_OPEN_GRIPPER_ING: # opening gripper
                state_cnt += 1
                if state_cnt * sim_node.config.decimation * sim_node.mj_model.opt.timestep > 0.5:
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(">>> state_sbb = {}".format(state_sbb))

            elif state_sbb == StateBuildBlocks.SBB_ARM_BACK:
                left_arm_cmd_buf[:6] = [0.0, -0.847,  1.2 ,  0.0, 1.5708, 0.88]
                right_arm_cmd_buf[:6] = [0.0, -0.847,  1.2 ,  0.0, -1.5708, -0.88]
                state_sbb = StateBuildBlocks(state_sbb.value+1)

            elif state_sbb == StateBuildBlocks.SBB_ARM_BACK_ING:
                if np.allclose(obs["jq"][12:18], left_arm_cmd_buf[:6], atol=1e-2) and np.allclose(obs["jq"][20:26], right_arm_cmd_buf[:6], atol=1e-2):
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_END:
                state_sbb = StateBuildBlocks.SBB_SLEEPING
                cube_idx += 1
                if cube_idx == len(cfg.obj_list):
                    input("Press Enter to continue...")
                    break

        except ValueError:
            traceback.print_exc()
            break

            state_cnt = 0
            action = sim_node.init_ctrl.copy()
            state_sbb = StateBuildBlocks.SBB_SLEEPING

            sim_node.reset()
            obs_lst, act_lst = [], []

        for i in range(2):
            chassis_cmd[i] = sim_node.step_func(chassis_cmd[i], chassis_cmd_buf[i], 2.0 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)

        for i in range(1):
            lift_cmd[i] = sim_node.step_func(lift_cmd[i], lift_cmd_buf[i], 5.0 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
        
        for i in range(2):
            head_cmd[i] = sim_node.step_func(head_cmd[i], head_cmd_buf[i], 5.0 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
        
        for i in range(7):
            if i == 2:
                left_arm_cmd[i]  = sim_node.step_func(left_arm_cmd[i] , left_arm_cmd_buf[i] , 2.4 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
                right_arm_cmd[i] = sim_node.step_func(right_arm_cmd[i], right_arm_cmd_buf[i], 2.4 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
            elif i == 6:
                left_arm_cmd[i]  = sim_node.step_func(left_arm_cmd[i] , left_arm_cmd_buf[i] , 10. * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
                right_arm_cmd[i] = sim_node.step_func(right_arm_cmd[i], right_arm_cmd_buf[i], 10. * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
            else:
                left_arm_cmd[i]  = sim_node.step_func(left_arm_cmd[i] , left_arm_cmd_buf[i] , 1.6 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
                right_arm_cmd[i] = sim_node.step_func(right_arm_cmd[i], right_arm_cmd_buf[i], 1.6 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)

        # obs_lst.append(obs)
        # act_lst.append(action.tolist())

        # if state_sbb > 10:
        #     save_path = os.path.join(save_dir, "{:03d}".format(data_idx))

        #     os.mkdir(save_path)
        #     with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        #         obj = {
        #             "time" : [o['time'] for o in obs_lst],
        #             "obs"  : {
        #                 "jq" : [o['jq'] for o in obs_lst],
        #             },
        #             "act"  : act_lst
        #         }
        #         json.dump(obj, fp)

        #     mediapy.write_video(os.path.join(save_path, "video.mp4"), [o['img'] for o in obs_lst], fps=cfg.render_set["fps"])

        #     data_idx += 1
        #     print("\r{:4}/{:4}".format(data_idx, data_set_size), end="")
            
        #     if data_idx >= data_set_size:
        #         break

        #     tarjq = np.zeros(6)
        #     state_sbb = 0
        #     sim_node.reset()

        #     obs_lst, act_lst= [], []

        obs, pri_obs, rew, ter, info = sim_node.step(action)

    print("")
