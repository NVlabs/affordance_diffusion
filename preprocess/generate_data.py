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
from argparse import ArgumentParser
import os
import os.path as osp
from typing import Tuple
from tqdm import tqdm
import pandas
from PIL import Image
import cv2
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

from sdedit import denoise_base, load_denoise_base
from sdedit import upsample as denoised_up
from sdedit import load_up as load_denoise_up

from dataset import HOI4D


has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')


def image2np(batch: th.Tensor):
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3]).detach().numpy()
    return reshaped


def read_image(path: str, size: int = 256) -> Tuple[th.Tensor, th.Tensor]:
    pil_img = Image.open(path).convert('RGB')
    pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1


def load_base():
    # Create base model.
    options = model_and_diffusion_defaults()
    options['inpaint'] = True
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    if args.base_ckpt is None:
        model.load_state_dict(load_checkpoint('base-inpaint', device))
    else:
        model.load_state_dict(th.load(args.base_ckpt))
    print('total base parameters', sum(x.numel() for x in model.parameters()))
    return model, diffusion, options


def load_up():
    # Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['inpaint'] = True
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint('upsample-inpaint', device))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))
    return model_up, diffusion_up, options_up


def predict_human(s=None,mask_file=None):
    if not osp.exists(mask_file):
        # deprecated: hand detection
        pass
    else:
        print('exist hand make', mask_file)
        ori_human_masks = cv2.imread(mask_file)
        masks = (ori_human_masks > 122.5)[..., 0]

    masks = Image.fromarray(masks.astype(np.uint8))
    masks = masks.resize((256, 256), resample=Image.BICUBIC)
    
    # Creating kernel
    if s is not None:
        kernel = np.ones((s, s), np.uint8)

        # Using cv2.erode() method 
        masks = cv2.erode(np.array(masks), kernel)

        masks = th.FloatTensor(masks[None, None])
    else:
        masks = None
    return masks, ori_human_masks


def load_input(fname, mask_file=None):
    # Source image we are inpainting
    pil_img = Image.open(fname).convert('RGB')
    source_image_256 = read_image(fname, size=256)
    source_image_64 = read_image(fname, size=64)

    source_mask_256, orig_mask = predict_human(args.kernel, mask_file=mask_file)
    source_mask_64 = F.interpolate(source_mask_256, (64, 64), mode='nearest')    
    return source_image_256, source_mask_256, source_image_64, source_mask_64, pil_img, orig_mask


##############################
# Sample from the base model #
##############################
def base_generate(model, diffusion, options, source_image_64, source_mask_64, prompt, ):
    batch_size = args.bs
    guidance_scale = args.scale # 5.0
    print('prompt: %s with scale %f' % (prompt, guidance_scale))

    # Create an classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    def denoised_fn(x_start):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
            x_start * (1 - model_kwargs['inpaint_mask'])
            + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
        )

    # Sampling parameters
    # Create the text tokens to feed to the model.
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens, options['text_ctx']
    )

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask=th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),

        # Masked inpainting image
        inpaint_image=(source_image_64 * source_mask_64).repeat(full_batch_size, 1, 1, 1).to(device),
        inpaint_mask=source_mask_64.repeat(full_batch_size, 1, 1, 1).to(device),
    )


    # Sample from the base model.
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]

    # Show the output
    # show_images(samples, '%s_outputx64' % index)
    return samples


##############################
# Upsample the 64x64 samples #
##############################
def upsample(model_up, diffusion_up, options_up, samples, source_image_256, source_mask_256, prompt, ):
    batch_size = args.bs
    upsample_temp = args.temp
    guidance_scale = args.scale

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

        # Masked inpainting image.
        inpaint_image=(source_image_256 * source_mask_256).repeat(batch_size, 1, 1, 1).to(device),
        inpaint_mask=source_mask_256.repeat(batch_size, 1, 1, 1).to(device),
    )

    def denoised_fn(x_start):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
            x_start * (1 - model_kwargs['inpaint_mask'])
            + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
        )

    # Sample from the base model.
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.p_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
        denoised_fn=denoised_fn,
    )[:batch_size]

    return up_samples

def dataloader(args):
    shuffle = True
    ds_list = []
    data_dir = args.data_dir
    save_dir = args.save_dir

    for split in args.split.split(','):
        ds = HOI4D(data_dir, save_dir, split, args.save_index)
        ds_list.append(ds)
    ds = ConcatDataset(ds_list)

    th.manual_seed(303)
    np.random.seed(303)
    dl = DataLoader(ds, None, True, num_workers=8, drop_last=False)
    return dl


def batch_main(args):
    dl = dataloader(args)

    glide = {}
    glide['denoise'] = load_denoise_base()
    glide['denoise_up'] = load_denoise_up()
    glide['base'] = load_base()
    glide['up'] = load_up()
    
    for i, data in tqdm(enumerate(dl), total=len(dl)):
        if data is None:
            continue
        if args.num > 0 and i > args.num:
            break
        if args.base_ckpt is not None:
            data['out_file'] = [osp.join(args.save_dir, osp.basename(data['out_file'][0]))]
        if args.skip and osp.exists(data['out_file']):
            print('skip', osp.exists(data['out_file']), data['out_file'])
            continue
        lock_file = data['out_file'] + '.lock'
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            if args.skip:
                continue
        try:
            inpaint_image(data['inp_file'], data['out_file'], glide, data)  # shall we do crop first?
        except Exception as e:
            if not KeyboardInterrupt and not args.debug:
                continue
            print(e)
            raise e
        os.system('rm -r %s' % lock_file)
        print('rm ', lock_file)


@th.no_grad()
def inpaint_image(inp_file, out_file, glide, data):
    prompt = data['prompt']

    source_image_256, source_mask_256, source_image_64, source_mask_64, ori_image, ori_mask \
        = load_input(inp_file, mask_file=data['mask_file'])

    if not args.dry:
        sample = base_generate(*glide['base'], source_image_64, source_mask_64, prompt)
        samplex256 = upsample(*glide['up'], sample, source_image_256, source_mask_256, prompt)

        out_image = Image.fromarray(image2np(samplex256))
        out_image = out_image.resize(ori_image.size)
    
        # save human mask as well~~
        if not osp.exists(data['mask_file']):
            os.makedirs(osp.dirname(data['mask_file']), exist_ok=True)
            print(data['mask_file'])
            ori_mask.save(data['mask_file'])

        os.makedirs(osp.dirname(out_file), exist_ok=True)
        print('save to ', out_file)
        out_image.save(out_file)

        inp_image = F.avg_pool2d(samplex256, 4)
        out_image64 = denoise_base(0.05, *glide['denoise'], inp_image, prompt)
        out_image = denoised_up(0, *glide['denoise_up'], out_image64, prompt)

        out_image = Image.fromarray(image2np(out_image))
        out_image = out_image.resize(ori_image.size)
        denoise_file = out_file.replace('glide_obj', 'denoised_obj')
        os.makedirs(osp.dirname(denoise_file), exist_ok=True)
        print(denoise_file)
        out_image.save(denoise_file)

    

def decode_one_vid(vid, image_dir):
    vid_dir = osp.join(image_dir, '{}', 'align_rgb/image.mp4')
    save_dir = osp.join(image_dir, '{}', 'align_frames')
    os.makedirs(save_dir.format(vid), exist_ok=True)

    cmd = 'rm -r %s/*' % save_dir.format(vid)
    print(cmd)
    os.system(cmd)
    
    cmd = "ffmpeg -hide_banner -loglevel fatal -i {} {}/%04d.png".format(vid_dir.format(vid), save_dir.format(vid))
    print(cmd)
    os.system(cmd) 


def decode_frame(csv_file, vid_index=None, rewrite=False):
    if vid_index is None:
        df = pandas.read_csv(csv_file)
        vid_index = list(set(df['vid_index']))
    save_dir = osp.join(args.data_dir,  'HOI4D_release', '{}', 'align_frames')
    for vid in tqdm(vid_index):
        if rewrite:
            os.system('rm -r %s' % save_dir.format(vid))
            decode_one_vid(vid, osp.join(args.data_dir, 'HOI4D_release'))
            print('rewrite')
        else:
            try:
                Image.open(save_dir.format(vid) + '/0001.png')
                print('continue', save_dir.format(vid), csv_file)
                continue
            except:
                print(save_dir.format(vid) + '/0001.png')
                lock_file = save_dir.format(vid) + '.lock'
                try:
                    os.makedirs(lock_file)
                except FileExistsError:
                    continue
                os.system('rm -r %s/*' % save_dir.format(vid))
                decode_one_vid(vid, osp.join(args.data_dir, 'HOI4D_release'))
                os.system('rm -r %s' % lock_file)



def parser_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/hoi4d/')
    parser.add_argument('--save_dir', type=str, default='output/tmp_hoi4d/')
    parser.add_argument('--save_index', type=str, default='HOI4D_glide')
    parser.add_argument('--split', type=str, default='docs/all_contact.csv')
    
    parser.add_argument('--base_ckpt', type=str, default=None)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--scale', type=float, default=5)
    parser.add_argument('--kernel', type=int, default=7)
    parser.add_argument('--temp', type=float, default=0.997)

    parser.add_argument('--num', type=int, default=-1)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--inpaint', action='store_true')
    parser.add_argument('--dir', type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parser_args()
    print('save to', args.save_dir )
    if args.decode:
        decode_frame(args.split)
    if args.inpaint:
        batch_main(args)
