'''
Part of the code (CUDA and OpenGL memory transfer) is derived from https://github.com/jbaron34/torchwindow/tree/master
'''
from OpenGL import GL as gl
import OpenGL.GL.shaders as shaders
import util
import util_gau
import numpy as np
import torch
from renderer_ogl import GaussianRenderBase
from dataclasses import dataclass
from cuda import cudart as cu
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

try:
    from OpenGL.raw.WGL.EXT.swap_control import wglSwapIntervalEXT
except:
    wglSwapIntervalEXT = None

@dataclass
class GaussianDataCUDA:
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-2]
    

@dataclass
class GaussianRasterizationSettingsStorage:
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool


def gaus_cuda_from_cpu(gau: util_gau) -> GaussianDataCUDA:
    gaus =  GaussianDataCUDA(
        xyz = torch.tensor(gau.xyz).float().cuda().requires_grad_(False),
        rot = torch.tensor(gau.rot).float().cuda().requires_grad_(False),
        scale = torch.tensor(gau.scale).float().cuda().requires_grad_(False),
        opacity = torch.tensor(gau.opacity).float().cuda().requires_grad_(False),
        sh = torch.tensor(gau.sh).float().cuda().requires_grad_(False)
    )
    gaus.sh = gaus.sh.reshape(len(gaus), -1, 3).contiguous()
    return gaus
    

class CUDARenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        self.raster_settings = {
            "image_height": int(h),
            "image_width": int(w),
            "tanfovx": 1,
            "tanfovy": 1,
            "bg": torch.Tensor([0., 0., 0]).float().cuda(),
            "scale_modifier": 1.,
            "viewmatrix": None,
            "projmatrix": None,
            "sh_degree": 3,  # ?
            "campos": None,
            "prefiltered": False,
            "debug": False
        }
        gl.glViewport(0, 0, w, h)
        self.program = util.load_shaders('shaders/gau_vert_cu.glsl', 'shaders/gau_frag_cu.glsl')

        # setup cuda
        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise RuntimeError(
                "OpenGL context may be running on integrated graphics"
            )
        
        self.vao = gl.glGenVertexArrays(1)
        self.tex = None
        self.set_gl_texture(h, w)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.need_rerender = True
        self.update_vsync()

    def update_vsync(self):
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)
        # else:
        #     print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.need_rerender = True
        if type(gaus) is dict:
            gau_xyz = []
            gau_rot = []
            gau_s = []
            gau_a = []
            gau_c = []
            for gaus_item in gaus.values():
                gau_xyz.append(gaus_item.xyz)
                gau_rot.append(gaus_item.rot)
                gau_s.append(gaus_item.scale)
                gau_a.append(gaus_item.opacity)
                gau_c.append(gaus_item.sh)
            gau_xyz = np.concatenate(gau_xyz, axis=0)
            gau_rot = np.concatenate(gau_rot, axis=0)
            gau_s = np.concatenate(gau_s, axis=0)
            gau_a = np.concatenate(gau_a, axis=0)
            gau_c = np.concatenate(gau_c, axis=0)
            gaus_all = util_gau.GaussianData(gau_xyz, gau_rot, gau_s, gau_a, gau_c)
            self.gaussians = gaus_cuda_from_cpu(gaus_all)
        else:
            self.gaussians = gaus_cuda_from_cpu(gaus)
        self.raster_settings["sh_degree"] = int(np.round(np.sqrt(self.gaussians.sh_dim))) - 1

    def sort_and_update(self, camera: util.Camera):
        self.need_rerender = True

    def set_scale_modifier(self, modifier):
        self.need_rerender = True
        self.raster_settings["scale_modifier"] = float(modifier)

    def set_render_mod(self, mod: int):
        self.need_rerender = True

    def set_gl_texture(self, h, w):
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            w,
            h,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(
            self.tex,
            gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to register opengl texture")
    
    def set_render_reso(self, w, h):
        self.need_rerender = True
        self.raster_settings["image_height"] = int(h)
        self.raster_settings["image_width"] = int(w)
        gl.glViewport(0, 0, w, h)
        self.set_gl_texture(h, w)

    def update_camera_pose(self, camera: util.Camera):
        self.need_rerender = True
        view_matrix = camera.get_view_matrix()
        view_matrix[[0,2], :] *= -1
        
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["viewmatrix"] = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings["campos"] = torch.tensor(camera.position).float().cuda()
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()

    def update_camera_pose_from_topic(self, camera: util.Camera, rmat, trans):
        self.need_rerender = True

        camera.position = np.array(trans).astype(np.float32)
        camera.target = camera.position - (1. * rmat[:3,2]).astype(np.float32)

        Tmat = np.eye(4)
        Tmat[:3,:3] = rmat
        Tmat[:3,3] = trans
        Tmat[0:3, [1,2]] *= -1
        transpose = np.array([[-1.0,  0.0,  0.0,  0.0],
                              [ 0.0, -1.0,  0.0,  0.0],
                              [ 0.0,  0.0,  1.0,  0.0],
                              [ 0.0,  0.0,  0.0,  1.0]])
        view_matrix = transpose @ np.linalg.inv(Tmat)

        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["viewmatrix"] = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings["campos"] = torch.tensor(camera.position).float().cuda()
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()

    def set_camera_pose(self, camera: util.Camera, rmat: np.array, trans: np.array):
        camera.position = np.array(trans).astype(np.float32)
        camera.target = camera.position - (1. * rmat[:3,2]).astype(np.float32)
        camera.is_pose_dirty = True

    def update_camera_intrin(self, camera: util.Camera):
        self.need_rerender = True
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()
        hfovx, hfovy, focal = camera.get_htanfovxy_focal()
        self.raster_settings["tanfovx"] = hfovx
        self.raster_settings["tanfovy"] = hfovy

    def draw(self):
        if self.reduce_updates and not self.need_rerender:
            gl.glUseProgram(self.program)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
            gl.glBindVertexArray(self.vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            return

        self.need_rerender = False

        # run cuda rasterizer now is just a placeholder
        raster_settings = GaussianRasterizationSettings(**self.raster_settings)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        # means2D = torch.zeros_like(self.gaussians.xyz, dtype=self.gaussians.xyz.dtype, requires_grad=False, device="cuda")
        with torch.no_grad():
            img, radii = rasterizer(
                means3D = self.gaussians.xyz,
                means2D = None,
                shs = self.gaussians.sh,
                colors_precomp = None,
                opacities = self.gaussians.opacity,
                scales = self.gaussians.scale,
                rotations = self.gaussians.rot,
                cov3D_precomp = None
            )

        img = img.permute(1, 2, 0)
        img = torch.concat([img, torch.ones_like(img[..., :1])], dim=-1)
        img = img.contiguous()
        height, width = img.shape[:2]
        # transfer
        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to map graphics resource")
        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to get mapped array")
        
        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            img.data_ptr(),
            4 * 4 * width,
            4 * 4 * width,
            height,
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to copy from tensor to texture")
        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to unmap graphics resource")

        gl.glUseProgram(self.program)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
