# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import cv2
import numpy as np
import torch
th = torch
import torch.nn as nn
import torch.nn.functional as F

def build_stn(cfg):
    if cfg.mask_mode == 'lollipop':
        net = LollipopMask(soft=cfg.soft_mask)
    return net


class LollipopMask(nn.Module):
    def __init__(self, hin=64, win=256, soft=False) -> None:
        super().__init__()
        self.inp_dim = 5
        
        self.hin = hin
        self.win = win
        self.soft = soft
        # creat a template
        template, ndcTtempl, templTloll = self.get_template(hin, win, soft)
        self.register_buffer('template', template)  
        self.register_buffer('ndcTloll', ndcTtempl@templTloll)
    
    @classmethod
    def get_template(self, hin, win, soft):
        size = hin // 2
        if not soft:
            template = np.zeros([hin, win])
            cv2.circle(template, (size, size), size, (1, 1, 1), -1)
            cv2.line(template, (size, size), (win, size), (1, 1, 1), size, -1)
            template = torch.FloatTensor(template)[None, None]  # (1, 1, hin, win)
        else:
            grid_coords = torch.stack(
                (torch.arange(win).repeat(hin), torch.arange(hin).repeat_interleave(win))).to(
                )  # [2, size*size]
            a = 0.83
            sigma = size/2*a
            dist = torch.exp(-0.5*(grid_coords[1] - hin/2)**2 / sigma**2) * (grid_coords[0] > size)
            dist1 = dist.reshape(1, 1, hin, win)

            sigma = size*a
            dist = (grid_coords[1] - hin/2)**2 + (grid_coords[0] - size)**2
            dist = torch.exp(-0.5 * (dist) / sigma ** 2)
            dist2 = dist.reshape(1, 1, hin, win)
            template = torch.max(dist1, dist2)

        ndcTtempl = self.ndcTpix(1, hin, win)
        templTloll = self.get_tempTloll(1, hin)
        return template, ndcTtempl, templTloll

    @classmethod
    def ndcTpix(self, N, H, W, device='cpu'):
        """(N, 3, 3)"""
        zeros = torch.zeros([N, 1], device=device)

        T = torch.stack([
            torch.cat([zeros+2/W, zeros, zeros-1], -1),
            torch.cat([zeros, zeros+2/H, zeros-1], -1),
            torch.cat([zeros, zeros, zeros+1], -1),
        ], 1)
        return T
    
    @classmethod
    def get_tempTloll(self, N, hin, win=None, device='cpu'):
        zeros = torch.zeros([N, 1], device=device)
        # templTloll = torch.stack([
        #     torch.cat([zeros+hin, zeros, hin/2+zeros], -1),
        #     torch.cat([zeros, zeros+hin, hin/2+zeros], -1),
        #     torch.cat([zeros, zeros, 1+zeros], -1),
        # ], 1)
        templTloll = torch.stack([
            torch.cat([zeros+hin, zeros, hin/2+zeros], -1),
            torch.cat([zeros, zeros+hin, hin/2+zeros], -1),
            torch.cat([zeros, zeros, 1+zeros], -1),
        ], 1)
        return templTloll

    @classmethod
    def pixTndc(self, N, H, W, device='cpu'):
        """(N, 3, 3)"""
        zeros = torch.zeros([N, 1], device=device)
        T = torch.stack([
            torch.cat([zeros+W/2, zeros, zeros+W/2], -1),
            torch.cat([zeros,zeros+ H/2, zeros+H/2], -1),
            torch.cat([zeros, zeros, zeros+1], -1),
        ], 1)
        return T

    def get_tempTcanvas(self, mask_param):
        x, y, sqrt_size, basis_i = mask_param.split([1, 1, 1, 2], dim=-1)
        scale = sqrt_size ** 2

        basis_i = F.normalize(basis_i, p=2, dim=-1)
        basis_j = torch.stack((-basis_i[..., 1], basis_i[..., 0]), -1)

        zeros = torch.zeros_like(x)  # (N, 1)
        S = torch.stack([
            torch.cat([1/scale, zeros, zeros], -1),  
            torch.cat([zeros, 1/scale, zeros], -1),  
            torch.cat([zeros, zeros, zeros+1], -1),  
        ], 1)  # (N, 3, 3)
        R = torch.stack([
            torch.cat([basis_i, zeros], -1),    # N, 3
            torch.cat([basis_j, zeros], -1),    # N, 3
            torch.cat([zeros, zeros, zeros+1], -1),  
        ], 1)
        T = torch.stack([
            torch.cat([zeros+1, zeros, -x], -1),  
            torch.cat([zeros, zeros+1, -y], -1),  
            torch.cat([zeros, zeros, zeros+1], -1),  
        ], 1)
        out = (S@R@T)
        return out

    def forward_image(self, mask_param, H, W=None, canvas_ndc=True, **kwargs):
        canvas = self(mask_param, H, W, canvas_ndc, **kwargs)
        out = {
            'mask': canvas,
            'image': canvas.repeat(1, 3, 1, 1)
        }
        return out

    def forward(self, mask_param, H, W=None, canvas_ndc=True, **kwargs):
        """
        :param mask_param: (N, 5), x,y in NDC space. sqrt(size), angle
        :param H: _description_
        """
        N = len(mask_param)
        device = mask_param.device
        if W is None:
            W = H
        tempTcanvas = self.get_tempTcanvas(mask_param)
        ndcTloll = self.ndcTloll.to(tempTcanvas)
        template = self.template.to(tempTcanvas)
        theta = ndcTloll @ tempTcanvas 
        if not canvas_ndc:
            canvasTndc = self.pixTndc(N, H, W, device)
            theta = theta @ canvasTndc

        # qtorch.Size([8, 3, 3]) 8 torch.Size([1, 3, 3]) torch.Size([8, 3, 3])
        grid = F.affine_grid(theta[:, :2], [N, 1, H, W])
        canvas = F.grid_sample(template.repeat(N, 1, 1, 1), grid)
        return  canvas

