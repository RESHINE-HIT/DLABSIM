import os
import cv2
import time
import mujoco
from scipy.spatial.transform import Rotation

import rospy
from geometry_msgs.msg import Pose, PoseArray

from dlabsim import DLABSIM_ASSERT_DIR
from dlabsim.viewer import ViewerBase, ViewerBaseConfig

class ViewerRos(ViewerBase):
    def __init__(self, config:ViewerBaseConfig):
        super().__init__(config)
        self.object_poses_suber = rospy.Subscriber('/object_poses', PoseArray, self.objectPosesCallback)

    def objectPosesCallback(self, msg:PoseArray):
        names = msg.header.frame_id.split(";")
        for name, pose in zip(names, msg.poses):
            pose:Pose = pose
            if name == "camera":
                pass
            else:
                try:
                    self.mj_model.body(name).pos = np.array([pose.position.x, pose.position.y, pose.position.z])
                    self.mj_model.body(name).quat = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
                    self.mj_data.body(name).xpos = self.mj_model.body(name).pos
                    self.mj_data.body(name).xquat = self.mj_model.body(name).quat
                except KeyError:
                    pass
        mujoco.mj_forward(self.mj_model, self.mj_data)

if __name__ == "__main__":
    import numpy as np
    np.set_printoptions(precision=3, suppress=True, linewidth=1000)

    rospy.init_node('Airbot_play_mujoco_viewer_node', anonymous=True)

    cfg = ViewerBaseConfig()
    cfg.robot = "airbot_play"
    cfg.mjcf_file_path = "mjcf/il-static.xml"
    cfg.render_set["fps"] = 30
    cfg.render_set["width"] = 1280
    cfg.render_set["height"] = 720
    cfg.put_text = False

    exec_node = ViewerRos(cfg)
    while exec_node.running:
        exec_node.render()

