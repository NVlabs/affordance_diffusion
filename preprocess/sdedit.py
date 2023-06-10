# wild mix of https://github.com/openai/glide-text2im and https://github.com/ermongroup/SDEdit
# MIT License

# Copyright (c) 2021 OpenAI https://github.com/openai/glide-text2im
# Copyright (c) 2021 Ermon Group https://github.com/ermongroup/SDEdit

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
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import torch
from tqdm import tqdm

import numpy as np
import torch as th

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
    SpacedDiffusion,
)

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')


def load_up():
    # Create base model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = False
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model, diffusion = create_model_and_diffusion(**options_up)

    model.eval()
    model.to(device)
    model.load_state_dict(load_checkpoint('upsample', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))
    return model, diffusion, options_up

def load_denoise_base():
    # Create base model.
    options = model_and_diffusion_defaults()
    options['inpaint'] = False
    options['use_fp16'] = False
    options['timestep_respacing'] = '1000' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    model.to(device)
    model.load_state_dict(load_checkpoint('base', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))
    return model, diffusion, options


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(diffusion: SpacedDiffusion, x, t, *,
                                               model,
                                               logvar,
                                               betas,model_kwargs,  **kwargs):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1 - diffusion.betas

    model_output = model(x, t, **model_kwargs)[:, :3]
    weighted_score = betas / diffusion.sqrt_one_minus_alphas_cumprod
    mean = extract(1 / np.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


def image_editing(diffusion:SpacedDiffusion, noise_level, x0, model,model_kwargs, **kwargs):
    e = torch.randn_like(x0)
    total_T = diffusion.num_timesteps  # TODO: pay attention to step -1 or not 0
    total_noise_levels = int(noise_level * total_T)  # 1: all noised  

    a = diffusion.alphas_cumprod
    # image_utils.save_images(x0, osp.join(args.save_dir, 'x0'), scale=True)
    x = x0 * diffusion.sqrt_alphas_cumprod[total_noise_levels - 1] + \
        e * diffusion.sqrt_one_minus_alphas_cumprod[total_noise_levels - 1]
    # image_utils.save_images(x, osp.join(args.save_dir, 'xT'), scale=True)
    n = 1

    posterior_variance = diffusion.posterior_variance
    logvar = np.log(np.append(posterior_variance[1], posterior_variance[1:]))

    # elif self.model_var_type == 'fixedsmall':
    #     self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    with tqdm(total=total_noise_levels) as progress_bar:
        for i in reversed(range(total_noise_levels)):
            t = (torch.ones(n) * i).to(device)
            
            x_ = image_editing_denoising_step_flexible_mask(diffusion, x, t=t, model=model,
                                                            logvar=logvar,
                                                            betas=diffusion.betas, model_kwargs=model_kwargs)
            x = x_
            progress_bar.update(1)
    
    x0 = x
    return x0


def upsample(ratio, model_up, diffusion_up, options_up, samples, prompt):
    batch_size = 1
    upsample_temp = 0.9

    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up['text_ctx']
    )

    # Create the model conditioning dict.
    model_kwargs = dict(
        # Low-res image to upsample.
        low_res=((samples+1)*127.5).round()/127.5 - 1,

        # Text tokens
        tokens=th.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )

    # Sample from the base model.
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.ddim_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )
    up_samples= up_samples[:batch_size]
    # image_utils.save_gif(sample_list, osp.join(args.save_dir, 'up'))
    return up_samples
    
    

def denoise_base(ratio, model, diffusion, options, image64, prompt):
    batch_size = 1

    # Sampling parameters
    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),

    )
    out = image_editing(diffusion, ratio, image64, model, model_kwargs)
    return out
