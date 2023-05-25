# https://github.com/afiaka87/glide-finetune
# MIT License

# Copyright (c) 2021 Clay Mullis

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

# --------------------------------------------------------
# Modified by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import os.path as osp
import numpy as np
import os
from typing import Tuple

import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch as th
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.bpe import Encoder

from models.hand_proxy import build_stn
from .train_utils import load_my_state_dict
from .train_utils import pred_to_pil
MODEL_TYPES = ["upsample", "base-inpaint", "upsample-inpaint"]


# TODO: clear !!!!

def get_uncond_tokens_mask(tokenizer: Encoder):
    uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask([], 128)
    return th.tensor(uncond_tokens), th.tensor(uncond_mask, dtype=th.bool)


def get_tokens_and_mask(
    tokenizer: Encoder, prompt: str = "", context_len: int = 128
) -> Tuple[th.tensor, th.tensor]:
    if len(prompt) == 0:
        return get_uncond_tokens_mask(tokenizer)
    else:
        tokens = tokenizer.encode(prompt)
        tokens, mask = tokenizer.padded_tokens_and_mask(tokens, context_len)
        tokens = th.tensor(tokens)  # + uncond_tokens)
        mask = th.tensor(mask, dtype=th.bool)  # + uncond_mask, dtype=th.bool)
        return tokens, mask


def load_model(
    glide_path: str = "",
    use_fp16: bool = False,
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    model_type: str = "base",
    module: str='',
    model_cls: str = 'InpaintText2ImUNet',
    model_kwargs={},
    cfg={},
):
    assert model_type in MODEL_TYPES, f"Model must be one of {MODEL_TYPES}. Exiting."
    if model_type in ["base", "base-inpaint"]:
        options = model_and_diffusion_defaults()
    elif model_type in ["upsample", "upsample-inpaint"]:
        options = model_and_diffusion_defaults_upsampler()
    if "inpaint" in model_type:
        options["inpaint"] = True
    options["use_fp16"] = use_fp16
    glide_model, glide_diffusion = create_model_and_diffusion(
        **options,  module=module, model_cls=model_cls, model_kwargs=model_kwargs)
    glide_model.splat_to_mask = build_stn(cfg)
    glide_model.requires_grad_(True)
    if freeze_transformer:
        glide_model.transformer.requires_grad_(False)
        glide_model.transformer_proj.requires_grad_(False)
        glide_model.token_embedding.requires_grad_(False)
        glide_model.padding_embedding.requires_grad_(False)
        glide_model.positional_embedding.requires_grad_(False)
    if freeze_diffusion:
        glide_model.out.requires_grad_(False)
        glide_model.input_blocks.requires_grad_(False)
        glide_model.middle_block.requires_grad_(False)
        glide_model.output_blocks.requires_grad_(False)
    # if glide_path is not None:
    #     if len(glide_path) > 0:  # user provided checkpoint
    #         print('train from ', glide_path)
    #         assert os.path.exists(glide_path), "pretrained path does not exist %s" % glide_path
    #         weights = th.load(glide_path, map_location="cpu")
    #         load_my_state_dict(glide_model, weights)
    #     else:  # use default checkpoint from openai
    #         print('use default')
    #         glide_model.load_state_dict(
    #             load_checkpoint(model_type, "cpu")
    #         )  # always load to cpu, saves memory
    # else:
    #     print('train from scratch')
    return glide_model, glide_diffusion, options

def read_image(path: str, shape: Tuple[int, int]):
    pil_img = PIL.Image.open(path).convert('RGB')
    pil_img = pil_img.resize(shape, resample=PIL.Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1

# Sample from the base model.

# for inpaint
@th.no_grad()
def sample(
    glide_model,
    glide_options,
    size,
    side_x=None,
    side_y=None,
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    prediction_respacing="100",
    upsample_enabled=False,
    inpaint_enabled=False,
    image_to_upsample='',
    upsample_temp=0.997,
    val_batch=None,
    soft_mask=False,
    hijack={},
    **kwargs,
):
    eval_diffusion = create_gaussian_diffusion(
        steps=glide_options["diffusion_steps"],
        noise_schedule=glide_options["noise_schedule"],
        timestep_respacing=prediction_respacing,
    )
    tokens, mask, reals, inpaint_image, inpaint_mask, mask_param,text = val_batch
    val_batch = tokens.to(device), mask.to(device), reals.to(device), \
            inpaint_image.to(device), inpaint_mask.to(device), mask_param.to(device), text
    tokens, mask, reals, inpaint_image, inpaint_mask, mask_param,text = val_batch

    # if soft_mask:
    inpaint_mask = 1 - glide_model.splat_to_mask(mask_param, size[-1], func_ab=lambda x: x**2)
    if not soft_mask:
        inpaint_mask = inpaint_mask > 0.5

    def cfg_model_fn(x_t, ts, **kwargs):
        # classifier-free
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = glide_model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        beta = eval_diffusion.betas[
            int(
                ts.flatten()[0].item()
                / glide_options["diffusion_steps"]
                * len(eval_diffusion.betas)
            )
        ]
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        current_prediction_pil = pred_to_pil(
            (x_t - eps * (beta**0.5))[:batch_size]
        )
        current_prediction_pil.save("current_prediction.png")
        return th.cat([eps, rest], dim=1)

    model_fn = cfg_model_fn # so we use CFG for the base model.
    if upsample_enabled:
        assert image_to_upsample != '', "You must specify a path to an image to upsample."
        low_res_samples = read_image(image_to_upsample, size=(side_x, side_y))
        model_kwargs['low_res'] = low_res_samples
        noise = th.randn((batch_size, 3, side_y, side_x), device=device) * upsample_temp
        model_kwargs['noise'] = noise
        model_fn = glide_model # just use the base model, no need for CFG.

    if inpaint_enabled:
        model_kwargs = dict(
                tokens=tokens,
                mask=mask,
            )
        noise = th.randn([batch_size, ] + size, device=device)
        input_kwargs = {
            'inpaint_image': inpaint_image,
            'inpaint_mask': inpaint_mask,
        }
        model_kwargs.update(input_kwargs)
        model_fn = glide_model # just use the base model, no need for CFG.
        full_batch_size = batch_size
   
    samples, sample_list = eval_diffusion.plms_sample_loop(
        model_fn,
        [full_batch_size,] + size,  
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        intermed=True,
        hijack=hijack,
    )
    return samples, sample_list 