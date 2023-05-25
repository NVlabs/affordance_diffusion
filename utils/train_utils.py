# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import re
import importlib
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import torch
th = torch
import torch.nn as nn
import logging

def pil_image_to_norm_tensor(pil_image):
    """
    Convert a PIL image to a PyTorch tensor normalized to [-1, 1] with shape [B, C, H, W].
    """
    return th.from_numpy(np.asarray(pil_image)).float().permute(2, 0, 1) / 127.5 - 1.0


def pred_to_pil(pred: th.Tensor, scale=True):
    if not scale:
        pred = pred * 2 - 1
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    ch = scaled.shape[1]
    if ch == 3:
        reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1, ch])
    else:
        reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1])
    return Image.fromarray(reshaped.numpy())



def to_cuda(data, device='cuda'):
    new_data = {}
    for key in data:
        if hasattr(data[key], 'cuda'):
            new_data[key] = data[key].to(device)
        else:
            new_data[key] = data[key]
    return new_data


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False        


def load_my_state_dict(model: nn.Module, state_dict, lambda_own=lambda x: x):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        own_name = lambda_own(name)
        # own_name = '.'.join(name.split('.')[1:])
        if own_name not in own_state:
            logging.warn('Model encounters unexpected param from checkpoint %s %s' % (name, own_name))
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if param.size() != own_state[own_name].size():
            logging.warn('size not match %s %s %s' % (
                name, str(param.size()), str(own_state[own_name].size())))
            continue
        own_state[own_name].copy_(param)
    
    for name in own_state:
        if name not in state_dict:
            logging.warn(f'Checkpoint misses key {name}')


def load_from_checkpoint(ckpt, cfg_file=None):
    if cfg_file is None:
        cfg_file = ckpt.split('checkpoints')[0] + '/config.yaml'
    print('use cfg file', cfg_file)
    cfg = OmegaConf.load(cfg_file)
    cfg.model.resume_ckpt = None  # save time to load base model :p
    module = importlib.import_module(cfg.model.module)
    model_cls = getattr(module, cfg.model.model)
    model = model_cls(cfg, )
    model.init_model()

    print('loading from checkpoint', ckpt)    
    weights = torch.load(ckpt)['state_dict']
    load_my_state_dict(model, weights)
    return model


def instantiate_from_config(config):
    # from ldm
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def ema_key_fix(key):
    # model_ema.diffusion_modeloutput_blocks71qkvbias
    key = key.replace('diffusion_model', 'diffusion_model.')
    key = key.replace('diffusion_model.out', 'diffusion_model.out.')
    key = key.replace('diffusion_model.out.put', 'diffusion_model.output')
    key = key.replace('blocks', 'blocks.')
    key = key.replace('middle_block', 'middle_block.')
    key = key.replace('layers', 'layers.')
    key = key.replace('norm', '.norm')
    key = key.replace('out_layers', '.out_layers')
    key = key.replace('in_layers', '.in_layers')
    key = key.replace('emb_layers', '.emb_layers')
    key = key.replace('skip_connection', '.skip_connection')
    key = key.replace('qkv', '.qkv')
    key = key.replace('proj_out', '.proj_out')
    key = key.replace('weight', '.weight')
    key = key.replace('bias', '.bias')
    key = key.replace('time_embed', 'time_embed.')
    # key = re.sub(r'out(?!\w)', 'out.', key)
    key = re.sub(r'(\D\d)(\d\D)', r'\1.\2', key)
    key = re.sub(r'(\D\d\d)(\d\D)', r'\1.\2', key)
    return key
