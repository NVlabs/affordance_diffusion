# --------------------------------------------------------
# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from glide_text2im.text2im_model import Text2ImUNet

from models.base import BaseModule
from utils.glide_utils import load_model
from utils.train_utils import load_my_state_dict


class InsertText2ImUNet(Text2ImUNet):
    """
    A text2im model which can perform inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2 + 1
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, inpaint_image=None, inpaint_mask=None, **kwargs):
        if inpaint_image is None:
            inpaint_image = torch.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = torch.zeros_like(x[:, :1])
        return super().forward(
            torch.cat([x, inpaint_image, inpaint_mask], dim=1),
            timesteps,
            **kwargs,
        )


class ContentNet(BaseModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.template_size = [3, cfg.side_y, cfg.side_x]
    
    def init_model(self,):
        cfg =self.cfg.model
        glide_model, glide_diffusion, glide_options = load_model(
            use_fp16=self.cfg.use_fp16,
            freeze_transformer=cfg.freeze_transformer,
            freeze_diffusion=cfg.freeze_diffusion,
            model_type='base-inpaint',        
            module=cfg.module,
            model_cls='InsertText2ImUNet',
            cfg=self.cfg,
        )
        self.glide_model = glide_model
        self.diffusion = glide_diffusion
        self.glide_options = glide_options

        ckpt_file = self.cfg.model.resume_ckpt
        if ckpt_file is not None:
            weights = torch.load(ckpt_file, map_location="cpu")
        else:
            return glide_model, glide_diffusion, glide_options
        if ckpt_file is not None:
            weights = torch.load(ckpt_file, map_location="cpu")
        if ckpt_file.endswith('.pt'):
            # if it's pre-trained glide:
            load_my_state_dict(self.glide_model, weights)
        elif ckpt_file.endswith('.ckpt'):
            weights = weights['state_dict']
            load_my_state_dict(self, weights)
        else:
            # scracth
            print('### train from scratch')
        return glide_model, glide_diffusion, glide_options

    def step(self, batch, batch_idx):
        device = self.device
        glide_model = self.glide_model
        glide_diffusion = self.diffusion

        tokens, masks, reals, inpaint_image, inpaint_mask, mask_param, _ = batch
        tokens = tokens.to(device)
        masks, reals, inpaint_image, inpaint_mask, mask_param, = masks.to(device), reals.to(device), inpaint_image.to(device), inpaint_mask.to(device), mask_param.to(device)
        if self.cfg.soft_mask:
            inpaint_mask = 1 - glide_model.splat_to_mask(mask_param, inpaint_mask.shape[-1], func_ab=lambda x: x**2)            
        timesteps = torch.randint(
            0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device
        )
        batch_size = len(masks)
        noise = torch.randn([batch_size,] + self.template_size, device=device)
        x_t = glide_diffusion.q_sample(reals, timesteps, noise=noise,
            ).to(device)
        model_output = glide_model(
            x_t.to(device),
            timesteps.to(device),
            mask=masks.to(device),
            tokens=tokens.to(device),
            inpaint_image=inpaint_image, 
            inpaint_mask=inpaint_mask, 
        )
        epsilon = model_output[:, :3]
        loss = F.mse_loss(epsilon, noise.to(device).detach())        
        return loss, {'loss': loss}

