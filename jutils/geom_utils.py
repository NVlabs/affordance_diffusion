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

import torch
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix, matrix_to_rotation_6d
import pytorch3d.transforms.rotation_conversions as rot_cvt


def axis_angle_t_to_matrix(axisang=None, t=None, s=None, homo=True):
    """
    :param axisang: (N, 3)
    :param t: (N, 3)
    :return: (N, 4, 4)
    """
    if axisang is None:
        axisang = torch.zeros_like(t)
    if t is None:
        t = torch.zeros_like(axisang)
    rot = rot_cvt.axis_angle_to_matrix(axisang)
    if homo:
        return rt_to_homo(rot, t, s)
    else:
        return rot

def matrix_to_se3(mat: torch.Tensor) -> torch.Tensor:
    """
    :param mat: transformation matrix in shape of (N, 4, 4)
    :return: tensor in shape of (N, 9) rotation param (6) + translation (3)
    """
    rot, trans, scale = homo_to_rt(mat)
    rot = matrix_to_rotation_6d(rot)
    se3 = torch.cat([rot, trans, scale], dim=-1)
    return se3


def se3_to_matrix(param: torch.Tensor):
    """
    :param param: tensor in shape of (..., 10) rotation param (6) + translation (3) + scale (1)
    :return: transformation matrix in shape of (N, 4, 4) sR+t
    """
    rot6d, trans, scale = torch.split(param, [6, 3, 3], dim=-1)
    rot = rotation_6d_to_matrix(rot6d)  # N, 3, 3
    mat = rt_to_homo(rot, trans, scale)    
    return mat


def rt_to_homo(rot, t=None, s=None):
    """
    :param rot: (..., 3, 3)
    :param t: (..., 3 ,(1))
    :param s: (..., 1)
    :return: (N, 4, 4) [R, t; 0, 1] sRX + t
    """
    rest_dim = list(rot.size())[:-2]
    if t is None:
        t = torch.zeros(rest_dim + [3]).to(rot)
    if t.size(-1) != 1:
        t = t.unsqueeze(-1)  # ..., 3, 1
    mat = torch.cat([rot, t], dim=-1)
    zeros = torch.zeros(rest_dim + [1, 4], device=t.device)
    zeros[..., -1] += 1
    mat = torch.cat([mat, zeros], dim=-2)
    if s is not None:
        s = scale_matrix(s)
        mat = torch.matmul(mat, s)

    return mat


def homo_to_rt(mat):
    """
    :param (N, 4, 4) [R, t; 0, 1]
    :return: rot: (N, 3, 3), t: (N, 3), s: (N, 1)
    """
    mat, _ = torch.split(mat, [3, mat.size(-2) - 3], dim=-2)
    rot_scale, trans = torch.split(mat, [3, 1], dim=-1)
    rot, scale = mat_to_scale_rot(rot_scale)

    trans = trans.squeeze(-1)
    return rot, trans, scale


def rt_to_homo(rot, t=None, s=None):
    """
    :param rot: (..., 3, 3)
    :param t: (..., 3 ,(1))
    :param s: (..., 1)
    :return: (N, 4, 4) [R, t; 0, 1] sRX + t
    """
    rest_dim = list(rot.size())[:-2]
    if t is None:
        t = torch.zeros(rest_dim + [3]).to(rot)
    if t.size(-1) != 1:
        t = t.unsqueeze(-1)  # ..., 3, 1
    mat = torch.cat([rot, t], dim=-1)
    zeros = torch.zeros(rest_dim + [1, 4], device=t.device)
    zeros[..., -1] += 1
    mat = torch.cat([mat, zeros], dim=-2)
    if s is not None:
        s = scale_matrix(s)
        mat = torch.matmul(mat, s)

    return mat


def mat_to_scale_rot(mat):
    """s*R to s, R
    
    Args:
        mat ( ): (..., 3, 3)
    Returns:
        scale: (..., 3)
        rot: (..., 3, 3)
    """
    sq = torch.matmul(mat, mat.transpose(-1, -2))
    scale_flat = torch.sqrt(torch.diagonal(sq, dim1=-1, dim2=-2))  # (..., 3)
    scale_inv = scale_matrix(1/scale_flat, homo=False)
    rot = torch.matmul(mat, scale_inv)
    return rot, scale_flat


def scale_matrix(scale, homo=True):
    """
    :param scale: (..., 3)
    :return: scale matrix (..., 4, 4)
    """
    dims = scale.size()[0:-1]
    one_dims = [1,] * len(dims)
    device = scale.device
    if scale.size(-1) == 1:
        scale = scale.expand(*dims, 3)
    mat = torch.diag_embed(scale, dim1=-2, dim2=-1)
    if homo:
        mat = rt_to_homo(mat)
    return mat

