import glfw
import imgui
import imageio
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer

import os
import numpy as np
import torch
import tkinter as tk
from pathlib import Path
from scipy.spatial.transform import Rotation

import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, PoseArray

import util
import util_gau
from renderer_ogl import OpenGLRenderer, GaussianRenderBase
from dlabsim import DLABSIM_ASSERT_DIR

BACKEND_OGL  = 0
BACKEND_CUDA = 1

def impl_glfw_init():
    global window
    window_name = "NeUVF editor"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(g_camera.w, g_camera.h, window_name, None, None)
    glfw.make_context_current(window)
    # glfw.swap_interval(1) #垂直同步
    glfw.swap_interval(0)  #无垂直同步
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)
    return window

def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False

def update_activated_renderer_state(gaus: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def main(models_set:set):
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_render_mode
    global g_pose_array_msg_recv

    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()
    root.withdraw()

    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    try:
        from renderer_cuda import CUDARenderer
        g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    except ImportError:
        g_renderer_idx = BACKEND_OGL
        print("CUDA renderer is not available.")
    else:
        g_renderer_idx = BACKEND_CUDA

    g_renderer = g_renderer_list[g_renderer_idx]

    # gaussian data
    gaussians_all = {}
    gaussians_idx = {}
    gaussians_size = {}

    idx_sum = 0
    gs_model_dir = Path(os.path.join(DLABSIM_ASSERT_DIR, "3dgs"))
    for model in models_set:
        data_path = Path(os.path.join(gs_model_dir, model))
        gs = util_gau.load_ply(data_path)
        gaussians_all[data_path.stem] = gs
        gaussians_idx[data_path.stem] = idx_sum
        gaussians_size[data_path.stem] = gs.xyz.shape[0]
        idx_sum += gaussians_size[data_path.stem]

    update_activated_renderer_state(gaussians_all)

    for name in gaussians_all.keys():
        # :TODO: 找到哪里被改成torch了
        try:
            gaussians_all[name].R = gaussians_all[name].R.numpy()
        except:
            pass

    while not glfw.window_should_close(window) and not rospy.is_shutdown():
        try:
            update_object_service = rospy.ServiceProxy('/update_object', Empty)
            response = update_object_service()
            break
        except rospy.ServiceException as e:
            rospy.logwarn_once("Service call failed: %s" % e)
            rospy.sleep(1.0)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        if g_pose_array_msg_recv:
            g_pose_array_msg_recv = False
            obj_names = g_pose_arr.header.frame_id.split(";")
            if len(obj_names):
                if len(obj_names) == len(g_pose_arr.poses):
                    update_gauss_data = False
                    for name, pose in zip(obj_names, g_pose_arr.poses):
                        trans = np.array([pose.position.x, pose.position.y, pose.position.z])
                        rmat = Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]).as_matrix()
                        if name in gaussians_all.keys():                            
                            if not ((gaussians_all[name].R == rmat).all() and (gaussians_all[name].T == trans).all()):
                                rospy.logwarn_once("Update pose of {}".format(name))
                                gaussians_all[name].T = trans
                                gaussians_all[name].R = rmat
                                gaussians_all[name].transform_by_matrix(rmat, trans)
                                g_renderer.gaussians.xyz[gaussians_idx[name]:gaussians_idx[name]+gaussians_size[name]] = gaussians_all[name].xyz_cu
                                g_renderer.gaussians.rot[gaussians_idx[name]:gaussians_idx[name]+gaussians_size[name]] = gaussians_all[name].rot_cu
                                update_gauss_data = True
                        elif name == "camera":
                            cam_rmat = np.eye(4)
                            cam_rmat[:3,:3] = rmat
                            cam_rmat[:3,3] = trans
                            g_renderer.update_camera_pose_from_topic(g_camera, cam_rmat[:3,:3], cam_rmat[:3,3])
                        # else:
                        #     rospy.logwarn("Invalid name: {}".format(name))
                    if update_gauss_data:
                        g_renderer.need_rerender = True
                else:
                    rospy.logwarn("Wrong size of poseArr len(name)={} while len(poseArr.poses)={}".format(len(obj_names), len(g_pose_arr.poses)))

        update_camera_intrin_lazy()
        
        g_renderer.draw()

        ##########################################################################################
        # width, height = glfw.get_framebuffer_size(window)
        # gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
        # gl.glReadBuffer(gl.GL_FRONT)
        # bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        # img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
        # imageio.imwrite("save-auto.png", img[::-1])
        ##########################################################################################

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)


    impl.shutdown()
    glfw.terminate()

def pose_array_callback(msg:PoseArray):
    global g_pose_array_msg_recv, g_pose_arr
    g_pose_array_msg_recv = True
    g_pose_arr = msg

if __name__ == "__main__":
    rospy.init_node('GaussianSplattingNode', anonymous=True)

    g_camera = util.Camera(1080, 1920)
    g_renderer_list = [None]
    g_renderer_idx = BACKEND_OGL
    g_renderer: GaussianRenderBase = g_renderer_list[g_renderer_idx]

    g_scale_modifier = 1.
    g_render_mode = 7

    g_pose_array_msg_recv = False
    g_pose_arr = PoseArray()
    g_pose_arr.header.frame_id = ""
    rospy.Subscriber('object_poses', PoseArray, pose_array_callback)

    models_set = {
        "qz11/il_env.ply",
        "qz11/cup_blue.ply",
        "qz11/cup_pink.ply",
        "airbot_play/arm_base.ply",
        "airbot_play/link1.ply",
        "airbot_play/link2.ply",
        "airbot_play/link3.ply",
        "airbot_play/link4.ply",
        "airbot_play/link5.ply",
        "airbot_play/link6.ply",
    }

    main(models_set)
