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

class MMK2Cfg(BaseConfig):
    expreriment    = "qiuzhi_11F_il"
    robot          = "mmk2"
    mjcf_file_path = "mjcf/mmk2_blocks_teleop.xml"
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
    init_joint_pose = {
        "base_position_x": 0.0,
        "base_position_y": 0.0,
        "base_position_z": 0.0,
        "base_orientation_w": 1.0,
        "base_orientation_x": 0.0,
        "base_orientation_y": 0.0,
        "base_orientation_z": 0.0,
        "lft_wheel": 0.0,  
        "rgt_wheel": 0.0,  
        "slide": 0.0,  
        "head_yaw": 0.0,  
        "head_pitch": 0.0,  
        "left_arm_joint1"  :  0.0,
        "left_arm_joint2"  :  0.0,
        "left_arm_joint3"  :  0.0,
        "left_arm_joint4"  :  0.0,
        "left_arm_joint5"  :  0.0,
        "left_arm_joint6"  :  0.0,
        "left_arm_gripper_0" :  0.0,
        "left_arm_gripper_1" :  0.0,
        "right_arm_joint1"  :  0.06382703,
        "right_arm_joint2"  : -0.71966516,
        "right_arm_joint3"  :  1.2772779,
        "right_arm_joint4"  : -1.5965166,
        "right_arm_joint5"  :  1.72517278,
        "right_arm_joint6"  :  1.80462028,
        "right_arm_gripper_0" :  1,
        "right_arm_gripper_1" :  1
    }
    """
    njqpos=28
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
        # TODO：初始位置配置
        # self.init_joint_pose = self.mj_model.key('home').qpos[:self.njq]
        self.init_joint_pose = np.array(list(config.init_joint_pose.values()))
        print(self.init_joint_pose)
        self.jq = np.zeros(self.njq)
        self.jv = np.zeros(self.njv)

        ip_cp = self.init_joint_pose.copy()
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
        # self.mj_data.qvel[:self.njv] = self.init_joint_pose[self.njq:self.njq+self.njq].copy()
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
    

    def __init__(self):
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
        self.subscription_encoder = self.create_subscription(
            Float32,
            'encoder_value',
            self.encoder_callback,
            10
        )

        self.x_value = 0.0
        self.y_value = 0.0
        self.button_value = 0.0
        self.encoder_value = 0.0

    def x_callback(self, msg):
        self.x_value = msg.data
        self.get_logger().info('X: "%s"' % msg.data)


    def y_callback(self, msg):
        self.y_value = msg.data
        self.get_logger().info('Y: "%s"' % msg.data)


    def button_callback(self, msg):
        self.button_value = msg.data
        self.get_logger().info(f'Received button_data: {msg.data}')

    def encoder_callback(self,msg):
        self.encoder_value = msg.data
        self.get_logger().info(f'Received encoder_value: {msg.data}')

    # def update_chassis_cmd(self):
    #     main.chassis_cmd[:] = [self.y_value, self.x_value]
        
def main(args=None):
    rclpy.init(args=args)
    exec_node = MMK2Base(MMK2Cfg())
   
    data_subscriber = MujocoRos2Node()

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

        #rclpy.spin_once(data_subscriber) #节点只运行一次！

        chassis_cmd[:]=[data_subscriber.x_value,data_subscriber.y_value]
        lift_cmd[:] =data_subscriber.button_value
        head_cmd[:]=[data_subscriber.encoder_value,1]
        obs, pri_obs, rew, ter, info = exec_node.step(action)#step运行仿真环境，运行一次！
        i += 1

    
    data_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

