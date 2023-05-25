# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import wandb
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from einops import rearrange

from glide_text2im.nn import timestep_embedding
from glide_text2im.text2im_model import Text2ImUNet
from glide_text2im.unet import TimestepEmbedSequential, AttentionBlock

from models.hand_proxy import build_stn
from models.base import BaseModule
from utils.train_utils import load_my_state_dict
from utils.glide_utils import load_model
from jutils import image_utils


class LayoutNet(BaseModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        if cfg.mask_mode == 'lollipop':
            lin_dim = 5
        elif cfg.mask_mode == 'pose':
            lin_dim = 3+45+6
        self.template_size = [lin_dim]
    
    def init_model(self):
        cfg = self.cfg.model
        glide_model, glide_diffusion, glide_options = load_model(
            use_fp16=self.cfg.use_fp16,
            freeze_transformer=cfg.freeze_transformer,
            freeze_diffusion=cfg.freeze_diffusion,
            model_type='base-inpaint',
            module=cfg.module,
            model_cls=cfg.get('target', 'SpatialUnet'),
            model_kwargs={'cfg': self.cfg},
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


    @torch.no_grad()
    def get_input(self, batch):
        device = self.device
        tokens, masks, reals, inpaint_image, inpaint_mask, mask_param, text = batch
        mask_param
        return {
            'x0': mask_param.to(device), 
            'tokens': tokens.to(device),
            'mask': masks.to(device),
            'inpaint_image': inpaint_image.to(device),
        }

    def step(self, batch, batch_idx):
        device = self.device
        glide_model = self.glide_model
        glide_diffusion = self.diffusion

        inp = self.get_input(batch)

        batch_size = len(inp['x0'])
        timesteps = torch.randint(
            0, len(glide_diffusion.betas) - 1, (batch_size,), device=device
        )
        noise = torch.randn([batch_size,] + self.template_size, device=device)
        x_t = glide_diffusion.q_sample(inp['x0'], timesteps, noise=noise,
            ).to(device)
        model_output = glide_model(
            x_t.to(device),
            timesteps.to(device),
            **inp,
        )
        epsilon = model_output
        loss, losses = 0, {}

        w = self.cfg.loss.naive
        if self.cfg.loss.naive_wd > 0:
            w *= float(self.global_step < self.cfg.loss.naive_wd)
        naive_loss = w * F.mse_loss(epsilon, noise.to(device).detach())
        losses['naive'] = naive_loss
        loss += naive_loss
        if self.cfg.loss.twod > 0:
            x_0_hat = self.diffusion.eps_to_pred_xstart(x_t, epsilon, timesteps)
            mask_pred = self.glide_model.splat_to_mask(x_0_hat,self.cfg.side_x, func_ab=lambda x: x**2)
            mask_gt = self.glide_model.splat_to_mask(inp['x0'], self.cfg.side_x, func_ab=lambda x: x**2)
            loss_2d = self.cfg.loss.twod * F.l1_loss(mask_pred, mask_gt.detach())
            losses['2d'] = loss_2d
            loss += loss_2d
        return loss, losses

    def vis_samples(self, batch, samples, sample_list, pref, log, step=None):
        # sample 
        obj_image = batch[3]
        mask = 1 - self.glide_model.splat_to_mask(samples, self.cfg.side_x, func_ab=lambda x: x**2)
        
        overlay = self.blend(mask.cpu(), obj_image.cpu())
        log['%ssample' % pref] = wandb.Image(vutils.make_grid(overlay))

        shape = samples.shape
        N = shape[0]
        sample_list = torch.stack(sample_list, 0).reshape(-1, *shape[1:])  # [T, N, xxxx]
        param_list = sample_list
        TN  = len(sample_list)
        T = TN // N
        _, _, H, W = batch[3].shape
        mask_list = 1 - self.glide_model.splat_to_mask(param_list, self.cfg.side_x, func_ab=lambda x: x**2)

        overlay_list = self.blend(mask_list.reshape(T, N, 1, H, W).cpu(), obj_image[None].repeat(T, 1, 1, 1, 1).cpu())
        fname = osp.join(self.logger.save_dir, '%05d_%s_progress') % (self.global_step, pref)
        image_utils.save_gif(overlay_list, fname)
        log['%sprogress' % pref] = wandb.Video(fname + '.gif')
        return 
    
    @torch.no_grad()
    def forward_w_gt(self, batch, gt_xy=True, gt_size=False):
        # others = batch[-1]
        mask_param = batch[-2]
        hij_param = torch.zeros_like(batch[-2])
        hij_mask = torch.zeros_like(batch[-2])
        hijack = {}
        if gt_xy:
            hij_param[...,0:2] =  mask_param[..., 0:2] # others['gt_xy'] 
            hij_mask[...,0:2] = 1
        if gt_size:
            hij_param[...,2:3] = mask_param[..., 2:3] # others['gt_size'] 
            hij_mask[...,2:3] = 1
        hijack = {
            'x0': hij_param,
            'mask': hij_mask,
        }
        return self.forward(batch, hijack=hijack)


class SpatialUnet(Text2ImUNet):
    """
    A inpainting model which can have explicit spatial mdoel?
    """

    def __init__(self, cfg, *args, **kwargs):
        # kwargs['encoder_channels'] = 768
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2 + 1
        super().__init__(*args, **kwargs)
        self.splat_to_mask = build_stn(cfg)
        self.init_model(cfg)
        mask_dim = self.splat_to_mask.inp_dim
        xf_width = kwargs.get('xf_width')  # 512?
        ch = 768 # self.middle_block.Ti  # TODO: dynamically 
        self.proj_in_param_img = nn.Linear(mask_dim, ch)
        self.spatial_img = TimestepEmbedSequential(
            AttentionBlock(
                ch,
                use_checkpoint=kwargs.get('use_checkpoint'),
                num_heads=kwargs.get('num_heads'),
                num_head_channels=kwargs.get('num_head_channels'),
                encoder_channels=ch,
            ),
        )

        self.spatial_txt = TimestepEmbedSequential(
            AttentionBlock(
                ch,
                use_checkpoint=kwargs.get('use_checkpoint'),
                num_heads=kwargs.get('num_heads'),
                num_head_channels=kwargs.get('num_head_channels'),
                encoder_channels=xf_width,
            ),
        )
        self.proj_out_param = nn.Linear(ch, mask_dim)

    def init_model(self, cfg):
        self.cfg = cfg
        self.lin_shape = [self.splat_to_mask.inp_dim]

    def concat(self, img, param):
        return param

    def split(self, img_param):
        return None, img_param

    def forward(self, img_mask, timesteps, inpaint_image=None, **kwargs):
        """

        :param img_mask: 2D vector of (N, 3*H*W+mask_dim)
        :param timesteps: _description_
        :param inpaint_image: _description_, defaults to None
        :return: 2D vector of (N, 3*H*W+mask_dim)
        """
        _, mask_param = self.split(img_mask)        
        device = mask_param.device
        if mask_param is None:
            mask_param = torch.zeros_like(mask_param)
            mask_param[:, 2:] = 1
        # TODO: the inpaint_mask may be completely outside of canvas? 
        inpaint_mask = 1 - self.splat_to_mask(mask_param, H=inpaint_image.shape[-1], func_ab=lambda x: x**2)
        if not self.cfg.soft_mask:
            inpaint_mask = (inpaint_mask > 0.5).float()
        if self.cfg.model.cond_mode == 'both':
            x = inpaint_image.to(device) * inpaint_mask.to(device) + (1 - inpaint_mask.to(device))
        elif self.cfg.model.cond_mode == 'learn':
            x = self.splat_to_mask.get_feat(mask_param, H=inpaint_image.shape[-1], func_ab=lambda x: x**2)
        else:
            raise NotImplementedError(self.cfg.cond_mode)
        mask = self._forward(
            torch.cat([x, inpaint_image, inpaint_mask], dim=1),
            timesteps,
            mask_param=mask_param,
            **kwargs,
        )
        out = self.concat(None, mask)
        return out
        
    def _forward(self, x, timesteps, tokens=None, mask=None, mask_param=None, **kwargs):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.xf_width:
            text_outputs = self.get_text_emb(tokens, mask)
            xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]
            # [8, 768], 512, 128
            emb = emb + xf_proj.to(emb)
        else:
            xf_out = None
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, xf_out)
            hs.append(h)
        # h emb torch.Size([8, 768, 8, 8]) torch.Size([8, 768]) torch.Size([8, 512, 128])
        h = self.middle_block(h, emb, xf_out)

        # TODO: add
        # emb torch.Size([8, 768, 8, 8]) torch.Size([8, 768])
        img_token = rearrange(h, 'n c h w -> n c (h w)')
        
        mask_h = self.spatial_img(
            self.proj_in_param_img(mask_param), emb, encoder_out=img_token)
        
        mask_h = self.spatial_txt(
            mask_h, emb, xf_out)
        mask_h = self.proj_out_param(mask_h)
        
        return mask_h

