# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import os.path as osp
import wandb
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from glide_text2im.tokenizer.bpe import get_encoder
from pytorch_lightning.utilities.distributed import rank_zero_only
from jutils import image_utils
from einops import rearrange
from utils.train_utils import ema_key_fix
from models.hand_proxy import build_stn
from models.base import BaseModule
from utils.train_utils import disabled_train, instantiate_from_config

from ldm.models.autoencoder import VQModelInterface
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    model_and_diffusion_defaults,
)


class CondModel(UNetModel):
    """
    A text2im model which can perform inpainting. # switch to ldm.openai backbone?
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            # kwargs["in_channels"] = kwargs["in_channels"] * 2 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1]
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, inpaint_image=None, inpaint_mask=None, **kwargs):
        if inpaint_image is None:
            inpaint_image = torch.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = torch.zeros_like(x[:, :1])
        out = super().forward(
            torch.cat([x, inpaint_image, inpaint_mask], dim=1),
            timesteps,
            **kwargs,
        )
        return out



class ContentNet(BaseModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.scale_factor = cfg.model.scale_factor
        self.cond_stage_forward = None
        
    def load_pretrain(self, ):
        if self.cfg.model.resume_ckpt is None:
            print('####### NO RESUME CKPT ####')
            return 
        sd = torch.load(self.cfg.model.resume_ckpt, map_location="cpu")['state_dict']
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            # unet to glide_model
            if k.startswith('model_ema.diffusion_model'):
                v = sd.pop(k)
                k = ema_key_fix(k)
                sd[k.replace('model_ema.diffusion_model', 'glide_model')] = v
        keys = list(sd.keys())
        if self.cfg.model.resume_ckpt is None:  # indicate we want train from scratch except for AE
            for k in keys:
                if 'glide_model' in k or 'cond_model' in k:
                    print('skip cond stage', k)
                    sd.pop(k)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    def set_template_size(self):
        cfg =self.cfg.model
        ch = cfg.unet_config.params.out_channels
        num_downs = self.first_stage_model.encoder.num_resolutions - 1
        rescale_latent = 2 ** (num_downs)
        size = cfg.unet_config.params.image_size = self.cfg.side_x // rescale_latent
        self.template_size = [ch, size, size]

    def instantiate_glide_model(self, cfg):
        self.set_template_size()
        self.glide_model = instantiate_from_config(cfg)

    def init_model(self,):
        cfg =self.cfg.model
        self.instantiate_first_stage(cfg.first_stage_config)
        self.instantiate_cond_stage(cfg.cond_stage_config)
        self.instantiate_glide_model(cfg.unet_config)
        self.load_pretrain()
        self.glide_model.tokenizer = get_encoder() # self.cond_stage_model.tokenizer
        self.glide_model.splat_to_mask = build_stn(self.cfg)

        glide_options = model_and_diffusion_defaults()        
        
        self.diffusion = create_gaussian_diffusion(
            steps=glide_options['diffusion_steps'],
            noise_schedule=glide_options['noise_schedule'],
            timestep_respacing=glide_options['timestep_respacing'],
        )
        self.eval_diffusion = create_gaussian_diffusion(
            steps=glide_options["diffusion_steps"],
            noise_schedule=glide_options["noise_schedule"],
            timestep_respacing='100',
        )
        self.glide_options = glide_options
        return self.glide_model, self.diffusion, glide_options

    @torch.no_grad()
    def get_input(self, batch):
        tokens, masks, reals, inpaint_image, _, mask_param, text = batch
        reals = self.encode(reals.to(self.device))    
        inpaint_mask = self.glide_model.splat_to_mask(mask_param.to(self.device), self.template_size[-1], func_ab=lambda x: x**2)
        if not self.cfg.soft_mask:
            inpaint_mask = inpaint_mask > 0.5

        inpaint_image = self.get_learned_conditioning(inpaint_image.to(self.device))  # (B, L, C) (batch_size, 77, 768)
        
        c = torch.cat([inpaint_image, inpaint_mask], 1)
        return {
            'x0': reals, 
            'text': text, 
            'inpaint_image': inpaint_image, 
            'inpaint_mask': inpaint_mask, 
            'context': c,
        }

    @torch.no_grad()
    def generate_sample_step(self, batch, pref, log, step=None, S=2):
        inp = self.get_input(batch)
        
        if step is None: step = self.global_step
        file_list = []
        step = self.global_step
        bs = len(inp['x0'])
        for n in range(S):
            samples, sample_list = self.eval_diffusion.plms_sample_loop(
                        self.glide_model,
                        [bs, ] + self.template_size,
                        device=self.device,
                        clip_denoised=True,
                        progress=True,
                        model_kwargs=inp,
                        cond_fn=None,
                        intermed=True,
                    )
            self.vis_samples(batch, samples, sample_list, pref + '%d_' % n, log, step)
        return file_list    

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__":
            print(f"Training {self.__class__.__name__} as an unconditional model.")
            self.cond_stage_model = None
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False


    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
    
    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def encode(self, x):
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        return z

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @rank_zero_only
    @torch.no_grad()
    def vis_input(self, batch, pref, log, step=None):
        log["%sgt" % pref] = wandb.Image(vutils.make_grid(batch[2]))
        overlay = self.blend(batch[4], batch[2])
        log["%smask" % pref] = wandb.Image(vutils.make_grid(overlay))
        log["%sobj" % pref] = wandb.Image(vutils.make_grid(batch[3]))

        obj = self.decode_first_stage(self.encode_first_stage(batch[3].to(self.device)))
        log["%sobj_recon" % pref] = wandb.Image(vutils.make_grid(obj))
        hoi = self.decode_first_stage(self.encode_first_stage(batch[2].to(self.device)))
        log["%shoi_recon" % pref] = wandb.Image(vutils.make_grid(hoi))

    @rank_zero_only
    def vis_samples(self, batch, samples, sample_list, pref, log, step=None):
        samples = self.decode_first_stage(samples)
        sample_list = [self.decode_first_stage(e) for e in sample_list]

        log['%ssample' % pref] = wandb.Image(vutils.make_grid(samples))
        fname = osp.join(self.logger.save_dir, '%05d_%s_progress') % (self.global_step, pref)
        image_utils.save_gif(sample_list, fname)
        log['%sprogress' % pref] = wandb.Video(fname + '.gif')

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
        losess = {}
        loss = self.cfg.loss.naive * F.mse_loss(epsilon, noise.to(device).detach())        
        losess['naive'] = loss
        return loss, losess        

    def forward(self, batch):
        inp = self.get_input(batch)
        bs = len(inp['x0'])

        samples, sample_list = self.eval_diffusion.plms_sample_loop(
                    self.glide_model,
                    [bs, ] + self.template_size,
                    device=self.device,
                    clip_denoised=True,
                    progress=True,
                    model_kwargs=inp,
                    cond_fn=None,
                    intermed=True,
                )
        samples = self.decode_first_stage(samples)
        sample_list = [self.decode_first_stage(e) for e in sample_list]
        return samples, sample_list