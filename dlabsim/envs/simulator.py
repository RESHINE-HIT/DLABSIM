import os
import cv2
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dlabsim import DLABSIM_ROOT_DIR, DLABSIM_ASSERT_DIR
from dlabsim.utils import JoyTeleop, BaseConfig, SingleObject, DLABSIM_KEY_DICT, DLABSIM_JOY_AVAILABLE

def setRenderOptions(options):
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    # options.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
    options.frame = mujoco.mjtFrame.mjFRAME_BODY.value
    pass

class SimulatorBase:
    running = True
    obs = None

    cam_id = -1
    render_cnt = 0
    camera_names = []
    mouse_last_x = 0
    mouse_last_y = 0

    sinobj_dict = {}

    camera_pose_changed = False
    camera_rmat = np.array([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  1,  0],
    ])

    options = mujoco.MjvOption()

    def __init__(self, config:BaseConfig):
        self.config = config

        if not self.config.headless:
            os.environ['MUJOCO_GL'] = 'egl'

        self.mjcf_file = os.path.join(DLABSIM_ASSERT_DIR, self.config.mjcf_file_path)
        if os.path.exists(self.mjcf_file):
            print("mjcf found: {}".format(self.mjcf_file))
        else:
            print("\033[0;31;40mFailed to load mjcf: {}\033[0m".format(self.mjcf_file))
            raise FileNotFoundError("Failed to load mjcf: {}".format(self.mjcf_file))

        self.mj_model = mujoco.MjModel.from_xml_path(self.mjcf_file)
        self.mj_model.opt.timestep = self.config.timestep
        self.mj_data = mujoco.MjData(self.mj_model)

        for i in range(self.mj_model.ncam):
            self.camera_names.append(self.mj_model.camera(i).name)

        assert -1 <= self.config.obs_camera_id < len(self.camera_names), "Invalid obs_camera_id {}".format(self.config.obs_camera_id)

        self.free_camera = mujoco.MjvCamera()
        self.free_camera.fixedcamid = -1
        self.free_camera.type = mujoco._enums.mjtCamera.mjCAMERA_FREE
        mujoco.mjv_defaultFreeCamera(self.mj_model, self.free_camera)

        self.renderer = mujoco.Renderer(self.mj_model, self.config.render_set["height"], self.config.render_set["width"])

        self.decimation = self.config.decimation
        self.delta_t = self.mj_model.opt.timestep * self.decimation
        self.render_fps = self.config.render_set["fps"]

        obj_names = self.mj_model.names.decode().split("\x00")
        for name in self.config.rb_link_list:
            if name in obj_names:
                self.sinobj_dict[name] = SingleObject(name, lazy_update=False)
            else:
                print("Invalid object name: {}".format(name))

        for name in self.config.obj_list:
            if name in obj_names:
                self.sinobj_dict[name] = SingleObject(name)
            else:
                print("Invalid object name: {}".format(name))

        mujoco.mjv_defaultOption(self.options)

        if DLABSIM_JOY_AVAILABLE and not self.config.headless and self.config.sync:
            try:
                self.teleop = JoyTeleop()
            except:
                self.teleop = None
        else:
            self.teleop = None

        if not self.config.headless:
            self.cv_windowname = self.config.robot.upper()
            cv2.namedWindow(self.cv_windowname, cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(self.cv_windowname, self.config.render_set["width"], self.config.render_set["height"])
            cv2.setMouseCallback(self.cv_windowname, self.mouseCallback)

        self.last_render_time = time.time()

    def __del__(self):
        print("**************************************************************************")
        print(">>>>>>>>>>>>>>>>>>>>> test del func of SimulatorBase <<<<<<<<<<<<<<<<<<<<<")
        self.renderer.close()
        if not self.config.headless:
            cv2.destroyAllWindows()
        print(">>>>>>>>>>>>>>>>>>>>> test del func of SimulatorBase <<<<<<<<<<<<<<<<<<<<<")
        print("**************************************************************************")

    def mouseCallback(self, event, x, y, flags, param):
        if self.config.mirror_image:
            x = self.config.render_set["width"] - x
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
        rgb_img = self.renderer.render()
        if cam_id == self.config.obs_camera_id:
            self.img_rgb_obs = rgb_img
        img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        if put_text:
            cv2.putText(img_bgr, "sim time: {:.3f}s".format(self.mj_data.time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img_bgr

    def cv2WindowKeyPressCallback(self, key):
        if key == -1:
            return True
        elif key == ord('h'):
            self.printHelp()
        elif key == ord("p"):
            self.printMessage()
        elif key == ord('q'):
            cv2.destroyWindow(self.cv_windowname)
            return False
        elif key == ord('r'):
            self.reset()
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
    
    def printHelp(self):
        print("Press 'h' to print help")
        print("Press 'q' to quit the program")
        print("Press 'r' to reset the state")
        print("Press '[' or ']' to switch camera view")
        print("Press 'Esc' to set free camera")
        print("Press 'p' to print the rotot state")

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time = {:.3f}".format(self.mj_data.time))
        print("mj_data.qpos = {}".format(np.array2string(self.mj_data.qpos, separator=', ')))
        print("mj_data.qvel = {}".format(np.array2string(self.mj_data.qvel, separator=', ')))
        print("mj_data.ctrl = {}".format(np.array2string(self.mj_data.ctrl, separator=', ')))
        print("-" * 100)

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.teleop:
            self.teleop.reset()

        mujoco.mj_forward(self.mj_model, self.mj_data)

    def getCameraPose(self):
        if self.cam_id == -1:
            rotation_matrix = self.camera_rmat @ Rotation.from_euler('xyz', [self.free_camera.elevation * np.pi / 180.0, self.free_camera.azimuth * np.pi / 180.0, 0.0]).as_matrix()
            camera_position = self.free_camera.lookat + self.free_camera.distance * rotation_matrix[:3,2]
        else:
            rotation_matrix = np.array(self.mj_data.camera(self.camera_names[self.cam_id]).xmat).reshape((3,3))
            camera_position = self.mj_data.camera(self.camera_names[self.cam_id]).xpos

        return Rotation.from_matrix(rotation_matrix).as_quat(), camera_position

    def getObjPose(self, name):
        try:
            position = self.mj_data.body(name).xpos
            quat = self.mj_data.body(name).xquat
            return position, quat
        except KeyError:
            print("Invalid object name: {}".format(name))
            return None, None

    def getObjectNameUpdate(self):
        obj_update_name_list = []
        for name in self.sinobj_dict.keys():
            position, quat = self.getObjPose(name)
            self.sinobj_dict[name].updatePose(position, quat)
            if not self.sinobj_dict[name].lazy_update or self.sinobj_dict[name].is_pose_dirty:
                obj_update_name_list.append(name)
                self.sinobj_dict[name].is_pose_dirty = False
        return obj_update_name_list

    def render(self):
        self.render_cnt += 1
        img_bgr_obs = self.getBgrImg(self.config.obs_camera_id, self.config.put_text)

        if self.cam_id == self.config.obs_camera_id:
            img_vis = img_bgr_obs
        else:
            img_vis = self.getBgrImg(self.cam_id, self.config.put_text)

        if not self.config.headless:
            if self.config.mirror_image:
                cv2.imshow(self.cv_windowname, cv2.flip(img_vis, 1))
            else:
                cv2.imshow(self.cv_windowname, img_vis)
            if self.config.sync:
                wait_time_ms = max(1, int((1./self.render_fps - time.time() + self.last_render_time) * 1000)-1)
            else:
                wait_time_ms = 1
            if not self.cv2WindowKeyPressCallback(cv2.waitKey(wait_time_ms)) or not cv2.getWindowProperty(self.cv_windowname, cv2.WND_PROP_VISIBLE):
                self.running = False
            if self.config.sync:
                self.last_render_time = time.time()

    # ------------------------------------------------------------------------------
    # ---------------------------------- Override ----------------------------------
    def reset(self):
        self.resetState()
        self.updateState()
        self.render()
        self.render_cnt = 0
        return self.getObservation()

    def updateState(self):
        pass

    def updateControl(self, action):
        pass
    
    def teleopProcess(self):
        if self.teleop.get_raising_edge(2): # "X"
            print("{} Sim Shutdown by JoyCmd".format(self.config["robot"]))
            self.running = False

    def getChangedObjectPose(self):
        raise NotImplementedError("pubObjectPose is not implemented")

    def checkTerminated(self):
        raise NotImplementedError("checkTerminated is not implemented")
    
    def post_physics_step(self):
        raise NotImplementedError("post_physics_step is not implemented")

    def getObservation(self):
        raise NotImplementedError("getObservation is not implemented")

    def getPrivilegedObservation(self):
        raise NotImplementedError("getPrivilegedObservation is not implemented")

    def getReward(self):
        raise NotImplementedError("getReward is not implemented")
    
    # ---------------------------------- Override ----------------------------------
    # ------------------------------------------------------------------------------

    def step(self, action=None):
        if self.teleop:
            self.teleopProcess()

        for _ in range(self.decimation):
            self.updateState()
            self.updateControl(action)
            mujoco.mj_step(self.mj_model, self.mj_data)

        if self.checkTerminated():
            self.resetState()
        
        self.post_physics_step()
        if self.render_cnt-1 < self.mj_data.time * self.render_fps:
            self.render()

        return self.getObservation(), self.getPrivilegedObservation(), self.getReward(), self.checkTerminated(), {}