import mujoco
import numpy as np

from dlabsim.utils.base_config import BaseConfig
from dlabsim.envs.simulator import SimulatorBase


import can  # python-can  4.3.1
import struct
import math
from std_msgs.msg import Float32
from std_msgs.msg import Int32

import rclpy
from rclpy.node import Node
''''''
class EncoderDriver(object):
    def __init__(self, motor_id, can_interface):
        self.inited = False
        self.board_id = motor_id
        self.can_interface = can_interface
        self.bus = can.interface.Bus(channel=can_interface, bustype="socketcan")
        self.ret_cmd_id = 0x588
        self.motor_pos = 0.0
        self.ret_id = 0
        self.notifier = can.Notifier(self.bus, [self._can_callback], timeout=1.0)
        self.inited = True
        self.x_value=None
        self.y_value=None

    # def __del__(self):
    #     if self.inited:
    #         self.notifier.stop()

    # def pos_cmd(self, pos, spd, ignore_limit):
    #     tx_frame = can.Message(
    #         arbitration_id=self.board_id, data=[0x07], is_extended_id=False
    #     )
    #     self._can_send(tx_frame)

    # def mit_cmd(self, f_p, f_v, f_kp, f_kd, f_t):
    #     tx_frame = can.Message(
    #         arbitration_id=self.board_id, dlc=0x01, data=[0x07], is_extended_id=False
    #     )
    #     self._can_send(tx_frame)

    # def set_zero(self):
    #     can_id = self.board_id | 0x80
    #     tx_frame1 = can.Message(
    #         arbitration_id=can_id, dlc=0x01, data=[0x0B], is_extended_id=False
    #     )
    #     self._can_send(tx_frame1)
    #     tx_frame2 = can.Message(
    #         arbitration_id=can_id, dlc=0x01, data=[0x0C], is_extended_id=False
    #     )
    #     self._can_send(tx_frame2)
    #     return True

    # def get_sSFtate(self):
    #     return self.motor_pos

    # def _can_send(self, tx_frame):
    #     self.bus.send(tx_frame)

    def _can_callback(self, msg: can.Message):
        if msg.arbitration_id == 0x522:  # is this motor
            data=msg.data[0]
            #print("Buttom value :",data)


        if msg.arbitration_id == 0x588:  # is this motor
            data=msg.data[0]

            if data & 0x80:  # 检查最高位是否为1
                    self.x_value = data & 0x7F  # 移除标志位
                    self.x_value = self.map_value(self.x_value, 93, True)
                    #print("X Value :", self.x_value)


            else :  # 最高位为0
                    self.y_value = data & 0x7F  # 移除标志位
                    self.y_value = self.map_value(self.y_value, 93, True)
                    #print("Y Value :", self.y_value)

            #mess=self.bus.recv()
            #print(mess)
        #print(msg.data[0])  
    def map_value(self, value, center, is_x):
        if value < center:
            mapped_value = (value - center) / center
        else:
            mapped_value = (value - center) / (127 - center)

        # 限幅算法：在输出绝对值小于0.1时输出为0
        if abs(mapped_value) < 0.2:
            mapped_value = 0.0

        return mapped_value              
''''''

class MMK2Cfg(BaseConfig):
    expreriment    = "qiuzhi_11F_il"
    robot          = "mmk2"
    mjcf_file_path = "mjcf/mmk2_floor.xml"
    timestep       = 0.0025
    decimation     = 4
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 30,
        # "width"  : 1280, #640, #1920 #1280
        # "height" : 720,  #480  #1080 #720
        # "width"  : 640,
        # "height" : 480 
        "width"  : 1920,
        "height" : 1080 
    }
    put_text       = True
    obs_camera_id  = 1
    rb_link_list   = []
    obj_list       = []
    """
    njqpos=29
    [0:7]-base; 7-lft_wheel; 8-rgt_wheel; 9-slide; 10-head_yaw"; 11-head_pitch; [12:20]-lft_arm ; [20:28]-rgt_arm

    njctrl=19
    0-forward; 1-turn; 2-lift; 3-yaw; 4-pitch; [5:12]-lft_arm; [12:19]-rgt_arm
    """

class MMK2Base(SimulatorBase):
    def __init__(self, config: MMK2Cfg):
        self.njq = 28
        self.njv = 27
        self.njctrl = 19

        super().__init__(config)
        self.init_joint_pose = self.mj_model.key('home').qpos[:self.njq]
        print(self.init_joint_pose)
        self.jq = np.zeros(self.njq)
        self.jv = np.zeros(self.njv)

        ip_cp = self.init_joint_pose.copy()
        # self.init_ctrl = np.zeros(19)
        self.init_ctrl = np.array(
            [0.0, 0.0] + 
            ip_cp[[9,10,11]].tolist() + 
            ip_cp[12:19].tolist() + 
            ip_cp[20:27].tolist()
        )

        self.resetState()

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.teleop:
            self.teleop.reset()

        self.jq = np.zeros(self.njq)
        self.jv = np.zeros(self.njv)

        self.mj_data.qpos[:self.njq] = self.init_joint_pose[:self.njq].copy()
        self.mj_data.ctrl[:self.njctrl] = self.init_ctrl.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateState(self):
        self.jq = self.mj_data.qpos[:self.njq]
        self.jv = self.mj_data.qvel[:self.njv]

    def updateControl(self, action):
        # if self.mj_data.qpos[12] < 0.0:
        #     self.mj_data.qpos[12] = 0.0
        # if self.mj_data.qpos[20] < 0.0:
        #     self.mj_data.qpos[20] = 0.0

        for i in range(self.njctrl):
            self.mj_data.ctrl[i] = action[i]
            # if i in {8, 10, 15, 17}:
            #     self.mj_data.ctrl[i] = self.step_func(self.mj_data.ctrl[i], action[i], 24 * self.mj_model.opt.timestep)
            # else:
            #     self.mj_data.ctrl[i] = action[i]

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
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time = {:.3f}".format(self.mj_data.time))
        print("mj_data.qpos :")
        print("    base      = {}".format(np.array2string(self.mj_data.qpos[:7], separator=', ')))
        print("    chassis   = {}".format(np.array2string(self.mj_data.qpos[7:9], separator=', ')))
        print("    lift      = {}".format(np.array2string(self.mj_data.qpos[9:10], separator=', ')))
        print("    head      = {}".format(np.array2string(self.mj_data.qpos[10:12], separator=', ')))
        print("    left  arm = {}".format(np.array2string(self.mj_data.qpos[12:19], separator=', ')))
        print("    right arm = {}".format(np.array2string(self.mj_data.qpos[20:27], separator=', ')))

        print("mj_data.qvel :")
        print("    base      = {}".format(np.array2string(self.mj_data.qvel[:6], separator=', ')))
        print("    chassis   = {}".format(np.array2string(self.mj_data.qvel[6:8], separator=', ')))
        print("    lift      = {}".format(np.array2string(self.mj_data.qvel[8:9], separator=', ')))
        print("    head      = {}".format(np.array2string(self.mj_data.qvel[9:11], separator=', ')))
        print("    left  arm = {}".format(np.array2string(self.mj_data.qvel[11:18], separator=', ')))
        print("    right arm = {}".format(np.array2string(self.mj_data.qvel[19:26], separator=', ')))

        print("mj_data.ctrl :")
        print("    chassis   = {}".format(np.array2string(self.mj_data.ctrl[0:2], separator=', ')))
        print("    lift      = {}".format(np.array2string(self.mj_data.ctrl[2:3], separator=', ')))
        print("    head      = {}".format(np.array2string(self.mj_data.ctrl[3:5], separator=', ')))
        print("    left  arm = {}".format(np.array2string(self.mj_data.ctrl[5:12], separator=', ')))
        print("    right arm = {}".format(np.array2string(self.mj_data.ctrl[12:19], separator=', ')))

        print("-" * 100)

class MujocoRos2Node(Node):
    def __init__(self, exec_node):
        super().__init__('mujoco_ros2_node')
        self.subscription_x = self.create_subscription(
            Float32,
            'x_value',
            self.x_callback,
            10
        )
        self.subscription_y = self.create_subscription(
            Float32,
            'y_value',
            self.y_callback,
            10
        )
        self.subscription_button = self.create_subscription(
            Int32,
            'button_data',
            self.button_callback,
            10
        )
        self.chassis_cmd = exec_node.action[:2]  # [forward, turn]
        self.exec_node = exec_node
        self.x_value = 0.0
        self.y_value = 0.0

    def x_callback(self, msg):
        self.x_value = msg.data

    def y_callback(self, msg):
        self.y_value = msg.data

    def button_callback(self, msg):
        self.get_logger().info(f'Received button_data: {msg.data}')

    def update_chassis_cmd(self):
        self.chassis_cmd[:] = [self.y_value, self.x_value]

if __name__ == "__main__":


    encoder_driver = EncoderDriver(0x588, "can0")
    encoder_driver = EncoderDriver(0x522, "can0")

    exec_node = MMK2Base(MMK2Cfg())

    obs = exec_node.reset()
    print(obs.keys())

    action = np.zeros(19)
    chassis_cmd = action[:2]  # [forward, turn]
    lift_cmd = action[2:3]  # [lift]
    head_cmd = action[3:5]  # [yaw, pitch]
    left_arm_cmd = action[5:12]  # [0, 0, 0, 0, 0, 0, eef]
    right_arm_cmd = action[12:19]  # [0, 0, 0, 0, 0, 0, eef]
    i = 0
    while exec_node.running:
        chassis_cmd[:] = [encoder_driver.y_value, encoder_driver.x_value]
        #print(encoder_driver.x_value[0])
        #print("y=",encoder_driver.y_value[1])
        #print(chassis_cmd[:])
        # chassis_cmd[0] = 1
        # action_list[5:11] = [i/200, i/200, i/200, i/200, i/200, i/200]
        obs, pri_obs, rew, ter, info = exec_node.step(action)
        i += 1