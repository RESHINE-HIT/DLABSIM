import os
import cv2
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dlabsim import DLABSIM_ASSERT_DIR

def setRenderOptions(options):
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
    options.frame = mujoco.mjtFrame.mjFRAME_BODY.value
    pass

class ViewerBaseConfig:
    robot          = "default"
    mjcf_file_path = ""
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,
        "height" :  720
    }
    put_text       = True

class ViewerBase:
    running = True

    cam_id = -1
    camera_names = []
    mouse_last_x = 0
    mouse_last_y = 0

    options = mujoco.MjvOption()

    def __init__(self, config:ViewerBaseConfig):
        self.config = config

        os.environ['MUJOCO_GL'] = 'egl'

        self.mjcf_file = os.path.join(DLABSIM_ASSERT_DIR, self.config.mjcf_file_path)
        if os.path.exists(self.mjcf_file):
            print("mjcf found: {}".format(self.mjcf_file))
        else:
            print("\033[0;31;40mFailed to load mjcf: {}\033[0m".format(self.mjcf_file))
            raise FileNotFoundError("Failed to load mjcf: {}".format(self.mjcf_file))

        self.mj_model = mujoco.MjModel.from_xml_path(self.mjcf_file)
        self.mj_data = mujoco.MjData(self.mj_model)

        for i in range(self.mj_model.ncam):
            self.camera_names.append(self.mj_model.camera(i).name)

        self.free_camera = mujoco.MjvCamera()
        self.free_camera.fixedcamid = -1
        self.free_camera.type = mujoco._enums.mjtCamera.mjCAMERA_FREE
        mujoco.mjv_defaultFreeCamera(self.mj_model, self.free_camera)

        self.renderer = mujoco.Renderer(self.mj_model, self.config.render_set["height"], self.config.render_set["width"])
        self.render_fps = self.config.render_set["fps"]

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        mujoco.mjv_defaultOption(self.options)

        self.cv_windowname = self.config.robot.upper() + "_VIEWER"
        cv2.namedWindow(self.cv_windowname, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.cv_windowname, self.config.render_set["width"], self.config.render_set["height"])
        cv2.setMouseCallback(self.cv_windowname, self.mouseCallback)

        self.last_render_time = time.time()

    def mouseCallback(self, event, x, y, flags, param):
        if self.cam_id == -1:
            action = None
            if flags == cv2.EVENT_FLAG_LBUTTON and event == cv2.EVENT_MOUSEMOVE:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
            elif flags == cv2.EVENT_FLAG_RBUTTON and event == cv2.EVENT_MOUSEMOVE:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif flags == cv2.EVENT_FLAG_MBUTTON and event == cv2.EVENT_MOUSEMOVE:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM
            if not action is None:
                self.camera_pose_changed = True
                _x, _y, _width, height = cv2.getWindowImageRect(self.cv_windowname)
                dx = float(x) - self.mouse_last_x
                dy = float(y) - self.mouse_last_y
                mujoco.mjv_moveCamera(self.mj_model, action, dx/height, dy/height, self.renderer.scene, self.free_camera)
        self.mouse_last_x = float(x)
        self.mouse_last_y = float(y)

    def getBgrImg(self, cam_id, put_text=True):
        if cam_id == -1:
            self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
        else:
            self.renderer.update_scene(self.mj_data, self.camera_names[cam_id], self.options)
        img_bgr = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        if put_text:
            cv2.putText(img_bgr, "sim time: {:.3f}s".format(self.mj_data.time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img_bgr

    def cv2WindowKeyPressCallback(self, key):
        if key == -1:
            return True
        elif key == ord('q'):
            cv2.destroyWindow(self.cv_windowname)
            return False
        elif key == 27: # "ESC"
            self.cam_id = -1
            self.camera_pose_changed = True
        elif key == ord(']') and self.mj_model.ncam:
            self.cam_id += 1
            self.cam_id = self.cam_id % self.mj_model.ncam
        elif key == ord('[') and self.mj_model.ncam:
            self.cam_id += self.mj_model.ncam - 1
            self.cam_id = self.cam_id % self.mj_model.ncam
        return True

    def render(self):
        img_vis = self.getBgrImg(self.cam_id, self.config.put_text)

        cv2.imshow(self.cv_windowname, img_vis)
        wait_time_ms = max(1, int((1./self.render_fps - time.time() + self.last_render_time) * 1000)-1)
        if not self.cv2WindowKeyPressCallback(cv2.waitKey(wait_time_ms)) or not cv2.getWindowProperty(self.cv_windowname, cv2.WND_PROP_VISIBLE):
            self.running = False

        self.last_render_time = time.time()

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=1000)

    cfg = ViewerBaseConfig()
    cfg.robot = "airbot_play"
    cfg.mjcf_file_path = "mjcf/il-static.xml"
    cfg.render_set["fps"] = 30
    cfg.render_set["width"] = 1280
    cfg.render_set["height"] = 720
    cfg.put_text = False

    exec_node = ViewerBase(cfg)
    while exec_node.running:
        exec_node.render()

