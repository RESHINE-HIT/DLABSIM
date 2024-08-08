import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
import pypose as pp
import torch

@dataclass
class GaussianData:
    def __init__(self, xyz, rot, scale, opacity, sh):
        self.xyz = xyz
        self.rot = rot
        self.scale = scale
        self.opacity = opacity
        self.sh = sh
        self.R = torch.zeros((3,3))
        self.T = np.zeros(3)

        num_points = self.xyz.shape[0]
        xyz_cu = torch.from_numpy(self.xyz).cuda()
        self.xyz_cu = torch.concatenate([xyz_cu, torch.ones((num_points, 1)).cuda()], dim=1).requires_grad_(False)
        self.rot_cu = torch.from_numpy(self.rot).cuda().requires_grad_(False)
        
        rot_xyzw_cu = torch.concatenate([self.rot_cu[:, 1:], self.rot_cu[:, 0].unsqueeze(1)], dim=1)
        trans_so3 = pp.SO3(rot_xyzw_cu)
        trans_matrix_cu = trans_so3.matrix()
        self.transMats_cu = torch.zeros(num_points, 4, 4).cuda().requires_grad_(False)
        self.transMats_cu[:, :3, :3] = trans_matrix_cu
        self.transMats_cu[:, :, 3] = self.xyz_cu

    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-1]

    def transform_by_matrix(self, R, T):
        num_points = self.xyz.shape[0]

        tmat = torch.zeros((4, 4)).cuda().requires_grad_(False)
        tmat[:3,:3] = torch.from_numpy(R).cuda().requires_grad_(False)
        tmat[:3, 3] = torch.from_numpy(T).cuda().requires_grad_(False)
        tmat[ 3, 3] = 1.0
        tmat = tmat.squeeze(0).repeat(num_points, 1, 1).requires_grad_(False)
        
        tmat_new = torch.bmm(tmat, self.transMats_cu).requires_grad_(False)
        self.xyz_cu = tmat_new[:, :3, 3]

        rot_new = tmat_new[:, :3, :3]
        quats_xyzw_new = pp.from_matrix(rot_new, ltype=pp.SO3_type, check=False).tensor().requires_grad_(False)
        self.rot_cu = torch.concatenate([quats_xyzw_new[:,3].unsqueeze(1), quats_xyzw_new[:, 0:3]], dim=1).requires_grad_(False)

def naive_gaussian():
    gau_xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0
    ]).astype(np.float32).reshape(-1, 4)
    gau_s = np.array([
        0.03, 0.03, 0.03,
        0.2, 0.03, 0.03,
        0.03, 0.2, 0.03,
        0.03, 0.03, 0.2
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = np.array([
        1, 0, 1, 
        1, 0, 0, 
        0, 1, 0, 
        0, 0, 1, 
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = (gau_c - 0.5) / 0.28209
    gau_a = np.array([
        1, 1, 1, 1
    ]).astype(np.float32).reshape(-1, 1)
    return GaussianData(
        gau_xyz,
        gau_rot,
        gau_s,
        gau_a,
        gau_c
    )


def load_ply(path):
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # pass activate function
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    opacities = 1/(1 + np.exp(- opacities))  # sigmoid
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([features_dc.reshape(-1, 3), 
                        features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)
    return GaussianData(xyz, rots, scales, opacities, shs)

if __name__ == "__main__":
    gs = load_ply("C:\\Users\\MSI_NB\\Downloads\\viewers\\models\\train\\point_cloud\\iteration_7000\\point_cloud.ply")
    a = gs.flat()
    print(a.shape)
