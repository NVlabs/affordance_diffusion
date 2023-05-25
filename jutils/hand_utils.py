# MIT License
#
# Copyright (c) 2018 The Python Packaging Authority
# Written by Yufei Ye (https://github.com/JudyYe)
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Modified from https://github.com/JudyYe/nnutils
# --------------------------------------------------------
from typing import Tuple
import torch
import torch.nn as nn
from pytorch3d.renderer import TexturesVertex, TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer import OrthographicCameras

from manopth.manolayer import ManoLayer
from . import geom_utils


class ManopthWrapper(nn.Module):
    def __init__(self, mano_path='third_party/mano', **kwargs):
        """
        :param mano_path: directory to MANO_RIGHT.pkl
        """
        super().__init__()
        self.mano_layer_right = ManoLayer(
            mano_root=mano_path, side='right', use_pca=kwargs.get('use_pca', False), ncomps=kwargs.get('ncomps', 45),
            flat_hand_mean=kwargs.get('flat_hand_mean', True))
        self.metric = kwargs.get('metric', 1)
        
        self.register_buffer('hand_faces', self.mano_layer_right.th_faces.unsqueeze(0))
        self.register_buffer('hand_mean', torch.FloatTensor(self.mano_layer_right.smpl_data['hands_mean']).unsqueeze(0))
        self.register_buffer('t_mano', torch.tensor([[0.09566994, 0.00638343, 0.0061863]], dtype=torch.float32, ))
        self.register_buffer('th_selected_comps', torch.FloatTensor(self.mano_layer_right.smpl_data['hands_components']))
        self.register_buffer('inv_scale', 1. / torch.sum(self.th_selected_comps ** 2, dim=-1))  # (D, ))

    def forward(self, glb_se3, art_pose, axisang=None, trans=None, return_mesh=True, 
        mode='outer', texture='verts', **kwargs) -> Tuple[Meshes, torch.Tensor]:
        N = len(art_pose)
        device = art_pose.device

        if mode == 'outer':
            if axisang is None:
                axisang = torch.zeros([N, 3], device=device)
            if trans is None:
                trans = torch.zeros([N, 3], device=device)
            if art_pose.size(-1) == 45:
                art_pose = torch.cat([axisang, art_pose], -1)
            verts, joints, faces = self._forward_layer(art_pose, trans)

            if glb_se3 is not None:
                if glb_se3.ndim == 3 and glb_se3.shape[-1] == 4:
                    mat_rt = glb_se3
                else:
                    mat_rt = geom_utils.se3_to_matrix(glb_se3)
                trans = Transform3d(matrix=mat_rt.transpose(1, 2))
                verts = trans.transform_points(verts)
                joints = trans.transform_points(joints)
        else:  # inner translation
            if axisang is None:
                axisang = torch.zeros([N, 3], device=device)
            art_pose = torch.cat([axisang, art_pose], -1)
            # if axisang is not None:
                # art_pose = torch.cat([axisang, art_pose], -1)
            if trans is None:
                trans = torch.zeros([N, 3], device=device)
            # if art_pose.size(-1) == 45:
            verts, joints, faces = self._forward_layer(art_pose, trans, **kwargs)

        if texture == 'verts':
            textures = torch.ones_like(verts)
            textures = TexturesVertex(textures)
        elif torch.is_tensor(texture):
            textures = TexturesUV(texture, self.faces_uv.repeat(N, 1, 1), self.verts_uv.repeat(N, 1, 1))

        else:
            raise NotImplementedError
        if return_mesh:
            return Meshes(verts, faces, textures), joints
        else:
            return verts, faces, textures, joints

    def _forward_layer(self, pose, trans, **kwargs):
        verts, joints = self.mano_layer_right(pose, th_trans=trans, **kwargs) # in MM
        verts /= (1000 / self.metric)
        joints /= (1000 / self.metric)

        faces = self.hand_faces.repeat(verts.size(0), 1, 1)

        return verts, joints, faces
    
    def mocap_to_wrapper(self, cam, pose, hack_z_clip=True):
        """ from 6+ncomps to hA + se3

        :param pose: (N, 6+ncomps) without mean_hand ? 
        :return: cTh in shape of (N, 4, 4)
        :return: hA in shape of (N, 45)
        """
        rot, pose = pose.split([3, pose.shape[-1] - 3], -1)
        # push further from camera to avoid z-clippi8ng
        N, device = len(pose), pose.device
        z_trans = torch.zeros([N, 3], device=device)
        if hack_z_clip:
            z_trans[..., -1] = 20
        cTh = geom_utils.axis_angle_t_to_matrix(rot, z_trans)
        # cTh = geom_utils.rt_to_homo(geom_utils.rotation_6d_to_matrix(so3), z_trans)
        
        hA = pose + self.hand_mean

        device = cam.device
        s, txty = cam.split([1, 2], -1)
        cameras = OrthographicCameras(s, s*txty, device=device)

        return cTh, hA, cameras
