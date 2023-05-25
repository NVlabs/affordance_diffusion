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
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import pickle
import torch
from pytorch3d.renderer.mesh.shader import HardPhongShader,  SoftPhongShader
import pytorch3d.structures.utils as struct_utils
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRasterizer, TexturesVertex, DirectionalLights, AmbientLights,
    RasterizationSettings, PerspectiveCameras, BlendParams,
)


def pad_texture(meshes: Meshes, feature: torch.Tensor='white') -> TexturesVertex:
    """
    :param meshes:
    :param feature: (sumV, C)
    :return:
    """
    if isinstance(feature, TexturesVertex):
        return feature
    if feature == 'white':
        feature = torch.ones_like(meshes.verts_padded())
    elif feature == 'blue':
        feature = torch.zeros_like(meshes.verts_padded())
        color = torch.FloatTensor([[[203,238,254]]]).to(meshes.device)  / 255   
        color = torch.FloatTensor([[[183,216,254]]]).to(meshes.device)  / 255 # * s - s/2
        feature = feature + color
    elif feature == 'yellow':
        feature = torch.zeros_like(meshes.verts_padded())
        # yellow = [250 / 255.0, 230 / 255.0, 154 / 255.0],        
        color = torch.FloatTensor([[[240 / 255.0, 207 / 255.0, 192 / 255.0]]]).to(meshes.device)
        color = color * 2 - 1
            # color = torch.FloatTensor([[[250 / 255.0, 230 / 255.0, 154 / 255.0]]]).to(meshes.device) * 2 - 1
        feature = feature + color
    elif feature == 'random':
        feature = torch.rand_like(meshes.verts_padded())  # [0, 1]
    if feature.dim() == 2:
        feature = struct_utils.packed_to_list(feature, meshes.num_verts_per_mesh().tolist())

    texture = TexturesVertex(feature)
    texture._num_faces_per_mesh = meshes.num_faces_per_mesh().tolist()
    texture._num_verts_per_mesh = meshes.num_verts_per_mesh().tolist()
    texture._N = meshes._N
    texture.valid = meshes.valid
    return texture


def render_mesh(meshes: Meshes, cameras, rgb_mode=True, depth_mode=False, **kwargs):
    """
    flip issue: https://github.com/facebookresearch/pytorch3d/issues/78
    :param meshes:
    :param out_size: H=W
    :param cameras:
    :param kwargs:
    :return: 'rgb': (N, 3, H, W). 'mask': (N, 1, H, W). 'rgba': (N, 3, H, W)
    """
    image_size = kwargs.get('out_size', 224)
    raster_settings = kwargs.get('raster_settings',
                                 RasterizationSettings(
                                     image_size=image_size, 
                                     faces_per_pixel=2,
                                     cull_backfaces=False))
    device = cameras.device

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)
    out = {}
    fragments = rasterizer(meshes, **kwargs)
    out['frag'] = fragments

    if rgb_mode:
        shader = kwargs.get('shader', HardPhongShader(device=meshes.device, lights=ambient_light(meshes.device, cameras, **kwargs)))
        image = shader(fragments, meshes, cameras=cameras, **kwargs)  # znear=znear, zfar=zfar, **kwargs)
        rgb, _ = flip_transpose_canvas(image)

        # get mask
        # Find out how much background_color needs to be expanded to be used for masked_scatter.
        N, H, W, K = fragments.pix_to_face.shape
        is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)
        alpha = torch.ones((N, H, W, 1), dtype=rgb.dtype, device=device)
        alpha[is_background] = 0.
        mask = flip_transpose_canvas(alpha, False)

        # Concat with the alpha channel.

        out['image'] = rgb
        out['mask'] = mask
    if depth_mode:
        zbuf = fragments.zbuf[..., 0:1]
        zbuf = flip_transpose_canvas(zbuf, False)
        out['depth'] = zbuf

    return out


def render_soft(meshes: Meshes, cameras: PerspectiveCameras, 
    rgb_mode=True, depth_mode=False, xy_mode=False, **kwargs):
    """
    :param meshes:
    :param cameras:
    :param kwargs:
    :return: 'image': (N, 3, H, W),
              'mask': (N, 1, H, W),
             'depth': (N, 1, H, W),
              'frag':,
    """
    blend_params = kwargs.get('blend_params', BlendParams(sigma=kwargs.get('sigma', 1e-5), gamma=1e-4))
    dist_eps = 1e-6
    # blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    # dist_eps = 1e-4
    device = cameras.device
    raster_settings = RasterizationSettings(
        image_size=kwargs.get('out_size', 224),
        blur_radius=np.log(1. / dist_eps - 1.) * blend_params.sigma,
        faces_per_pixel=kwargs.get('faces_per_pixel', 10),
        perspective_correct=False,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)

    out = {}
    fragments = rasterizer(meshes, **kwargs)
    out['frag'] = fragments

    if rgb_mode:
        # shader = SoftGouraudShader(device, lights=ambient_light(meshes.device, cameras))
        shader = kwargs.get('shader', SoftPhongShader(device, lights=ambient_light(meshes.device, cameras, **kwargs)))
        if torch.isnan(fragments.zbuf).any():
            fname = '/checkpoint/yufeiy2/hoi_output/vis/mesh.pkl'
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fp:
                pickle.dump({'mesh': meshes, 'camera': cameras}, fp)
            import pdb
            pdb.set_trace()
        image = shader(fragments, meshes, cameras=cameras, blend_params=blend_params)  # znear=znear, zfar=zfar, **kwargs)
        rgb, mask = flip_transpose_canvas(image)

        out['image'] = rgb
        out['mask'] = mask

    if depth_mode:
        zbuf = fragments.zbuf[..., 0:1]
        zbuf = flip_transpose_canvas(zbuf, False)
        zbuf[zbuf != zbuf] = -1
        out['depth'] = zbuf

    if xy_mode:
        cVerts = meshes.verts_padded()
        iVerts = cameras.transform_points_ndc(cVerts)
        # iVerts[..., 1] *= -1
        out['xy'] = iVerts
    return out


def flip_transpose_canvas(image, rgba=True):
    image = torch.flip(image, dims=[1, 2])  # flip up-down, and left-right
    image = image.transpose(-1, -2).transpose(-2, -3)  # H, 4, W --> 4, H, W
    if rgba:
        rgb, mask = torch.split(image, [image.size(1) - 1, 1], dim=1)  # [0-1]
        return rgb, mask
    else:
        return image


def ambient_light(device='cpu', cameras: PerspectiveCameras = None, **kwargs):
    d = torch.FloatTensor([[0, 0, -1]]).to(device)
    N = 1 if cameras is None else len(cameras)
    zeros = torch.zeros([N, 3], device=device)
    d = zeros + d
    if cameras is not None:
        d = cameras.get_world_to_view_transform().inverse().transform_normals(d.unsqueeze(1))
        d = d.squeeze(1)

    color = kwargs.get('light_color', np.array([0.65, 0.3, 0.0]))
    D = kwargs.get('dims', 3)
    t_zeros = torch.zeros([N, D], device=device)
    if D == 3:
        am, df, sp = color
        am = t_zeros + am
        df = t_zeros + df
        sp = t_zeros + sp
        lights = DirectionalLights(
            device=device,
            ambient_color=am,
            diffuse_color=df,
            specular_color=sp,
            direction=d,
        )
    else:
        lights = MyAmbientLights(ambient_color=t_zeros + 1, device=device)
    return lights



class MyAmbientLights(AmbientLights):
    def __init__(self, *, ambient_color=None, device = "cpu") -> None:
        super().__init__(ambient_color=ambient_color, device=device)
        self.D = self.ambient_color.shape[-1]

    def diffuse(self, normals, points) -> torch.Tensor:
        N = len(points)
        D = self.D
        return torch.zeros([N, D], device=self.device)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        N = len(points)
        D = self.D
        return torch.zeros([N, D], device=self.device)    
