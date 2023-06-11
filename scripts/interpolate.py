# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

"""
Usage: 
- Run demo: 
python -m scripts.interpolate dir=docs/demo_inter output=$path_to_model_dir \

- Run on predicted parameters: 
Assume you have run python inference.py --config-name=test  to get initial layout parameter precdition
Then run 
python -m scripts.interpolate \
  dir=\${output}/release/layout/cascade \ # path that saves predicted layout param
"""
import pickle
import imageio
import os
import os.path as osp
from hydra import main
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import torch
from jutils import image_utils
from glide_text2im.tokenizer.bpe import get_encoder
from inference import CascadeAffordance


def get_mask_list(init, num, orient=True):
    oppo = init.clone()
    oppo[..., 0] *= -1

    alpha = torch.linspace(0, 1, num).unsqueeze(-1)  # (S, 1)
    mask_params = alpha * oppo + (1 - alpha) * init  # 

    if orient is True:
        th0 = torch.atan2(init[..., -2], init[..., -1])
        th1 = torch.atan2(-init[..., -2], init[..., -1]) 
        th_all = th1 * alpha + th0 * (1 - alpha)
        e12 = torch.cat([torch.sin(th_all), torch.cos(th_all)], -1)
        mask_params[..., -2:] = e12
    return mask_params


def read_mask_param(fname, test_num, orient=True):
    init = pickle.load(open(fname, 'rb'))
    init = torch.FloatTensor(init)[None]
    mask_params = get_mask_list(init, test_num, orient)
    return mask_params


def read_image(fname):
    image = imageio.imread(fname)
    image = ToTensor()(image) * 2 - 1
    return image


def make_batch(image, mask_param, tokenizer, device):
    bs = len(mask_param)
    obj_image = image[None].repeat(bs, 1,1,1).to(device)
    
    uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask([], 128)
    uncond_tokens = torch.FloatTensor(uncond_tokens)[None].repeat(bs, 1).to(device)
    uncond_mask = torch.BoolTensor(uncond_mask)[None].repeat(bs, 1).to(device)

    obj_mask = torch.ones_like(obj_image[:, 0:1])
    batch = uncond_tokens, uncond_mask, obj_image, obj_image, obj_mask, mask_param, ['']
    return batch


@torch.no_grad()
@main('../configs', 'test', version_base=None)
def main_worker(cfg):
    pl.seed_everything(123)

    test_name = cfg.interpolation.test_name
    wrapper = CascadeAffordance(cfg.what_ckpt, cfg.where_ckpt, cfg=cfg, save_dir=cfg.dir)
    cfg.dir = wrapper.save_dir

    name = cfg.interpolation.index
    
    if not cfg.dry:
        wrapper.init_model()
        # read mask param
        fname = osp.join(wrapper.save_dir, 'mask_param', name +'.pkl')
        mask_params = read_mask_param(fname, cfg.interpolation.len, orient=cfg.interpolation.orient)
        image = read_image(osp.join(wrapper.save_dir, 'inp', name[:-2]+'.png'), )
        batch = make_batch(image, mask_params, get_encoder(), wrapper.device)
        image_batch = batch[3]
        # save input 
        wrapper.save_batch_images(image_batch, osp.join(test_name, 'inp'), name)
        print(os.getcwd(), osp.join(wrapper.save_dir, test_name, 'inp'))
        for s in range(cfg.interpolation.num):
            sample, _, _ = wrapper(batch, True)
            # save output
            wrapper.save_batch_images(sample, osp.join(test_name, 'overall'), name, 's%d' % s)

            masks = wrapper.splat_to_mask(mask_params, image.shape[-1])
            overlay = image_utils.blend_images(torch.ones_like(image_batch), image_batch, masks, 0.7)
            # save input mask
            wrapper.save_batch_images(overlay, osp.join(test_name, 'mask'), name, 's%d' % s)
        wrapper.superres_eval(osp.abspath(osp.join(wrapper.save_dir, test_name, 'overall')))

    make_gif(name + '_00_s0', osp.join(wrapper.save_dir, test_name), cfg.interpolation.len)


def make_gif(start_frame, data_dir, T):
    pref = '_'.join(start_frame.split('_')[:-2])
    suf = start_frame.split('_')[-1]

    image_list = []
    mask_list = []
    for t in range(T):
        index = f'{pref}_%02d_{suf}'%(t)
        overall_file = osp.join(data_dir, 'superres', index + '.png')
        mask_file = osp.join(data_dir, 'mask', index + '.png')

        image_list.append(imageio.imread(overall_file))
        mask_list.append(imageio.imread(mask_file))
    t = 10
    imageio.mimsave(osp.join(data_dir, index) + '_image.gif', image_list, duration=1./t)
    imageio.mimsave(osp.join(data_dir, index) + '_mask.gif', mask_list, duration=1./t)
    print('saved to', osp.join(data_dir, index) + '_image.gif')


if __name__ == '__main__':
    main_worker()