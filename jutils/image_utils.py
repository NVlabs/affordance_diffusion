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

import os
import os.path as osp
from PIL import Image
import cv2
import imageio
import numpy as np
import torch
import torchvision.utils as vutils


def crop_weak_cam(cam, bbox_topleft, oldo2n, 
    new_center, new_size, old_size=224, resize=224):
    """
    Args:
        cam ([type]): [description]
        bbox_topleft ([type]): [description]
        scale ([type]): [description]
        new_bbox ([type]): [description] 
    """
    cam = cam.copy()
    s, t = np.split(cam, [1, ], -1)
    prev_center = bbox_topleft + (old_size / 2) / oldo2n
    offset = (prev_center - new_center)

    newo2n = resize/new_size
    
    # t += offset / (resize / 2) / s  * oldo2n
    s *=  newo2n / oldo2n * old_size / resize
    t += 2 *  newo2n * offset / resize / s
    new_cam = np.concatenate([s, t], -1)

    new_tl = new_center - new_size / 2
    new_scale = newo2n
    return new_cam, new_tl, new_scale


def save_images(images, fname, text_list=[None], merge=1, col=8, scale=False, bg=None, mask=None, r=0.9,
                keypoint=None, color=(0, 1, 0)):
    """
    :param it:
    :param images: Tensor of (N, C, H, W)
    :param text_list: str * N
    :param name:
    :param scale: if RGB is in [-1, 1]
    :param keypoint: (N, K, 2) in scale of [-1, 1]
    :return:
    """
    if bg is not None:
        images = blend_images(images, bg, mask, r)

    merge_image = tensor_text_to_canvas(images, text_list, col=col, scale=scale)

    if fname is not None:
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        imageio.imwrite(fname + '.png', merge_image)
    return merge_image


def blend_images(fg, bg, mask=None, r=0.9):
    fg = fg.cpu()
    bg=bg.cpu()
    if mask is None:
        image = fg.cpu() * r + bg.cpu() * (1-r)
    else:
        mask = mask.cpu().float()
        image = bg * (1 - mask) + (fg * r + bg * (1 - r)) * mask
    return image


def save_gif(image_list, fname, text_list=[None], merge=1, col=8, scale=True):
    """
    :param image_list: [(N, C, H, W), ] * T
    :param fname:
    :return:
    """

    def write_to_gif(gif_name, tensor_list, batch_text=[None], col=8, scale=False):
        """
        :param gif_name: without ext
        :param tensor_list: list of [(N, C, H, W) ] of len T.
        :param batch_text: T * N * str. Put it on top of
        :return:
        """
        T = len(tensor_list)
        if batch_text is None:
            batch_text = [None]
        if len(batch_text) == 1:
            batch_text = batch_text * T
        image_list = []
        for t in range(T):
            time_slices = tensor_text_to_canvas(tensor_list[t], batch_text[t], col=col,
                                                scale=scale)  # numpy (H, W, C) of uint8
            image_list.append(time_slices)
        # write_mp4(image_list, gif_name)
        write_gif(image_list, gif_name)
    # merge write
    if len(image_list) == 0:
        print('not save empty gif list')
        return
    num = image_list[0].size(0)
    if merge >= 1:
        write_to_gif(fname, image_list, text_list, col=min(col, num), scale=scale)
    if merge == 0 or merge == 2:
        for n in range(num):
            os.makedirs(fname, exist_ok=True)
            single_list = [each[n:n+1] for each in image_list]
            write_to_gif(os.path.join(fname, '%d' % n), single_list, [text_list[n]], col=1, scale=scale)

def write_gif(image_list, gif_name):
    if not os.path.exists(os.path.dirname(gif_name)):
        os.makedirs(os.path.dirname(gif_name))
        print('## Make directory: %s' % gif_name)
    imageio.mimsave(gif_name + '.gif', image_list)
    print('save to ', gif_name + '.gif')


def tensor_text_to_canvas(image, text=None, col=8, scale=False):
    """
    :param image: Tensor / numpy in shape of (N, C, H, W)
    :param text: [str, ] * N
    :param col:
    :return: uint8 numpy of (H, W, C), in scale [0, 255]
    """
    if scale:
        image = image / 2 + 0.5
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    image = image.cpu().detach()  # N, C, H, W

    image = write_text_on_image(image.numpy(), text)  # numpy (N, C, H, W) in scale [0, 1]
    image = vutils.make_grid(torch.from_numpy(image), nrow=col)  # (C, H, W)
    image = image.numpy().transpose([1, 2, 0])
    image = np.clip(255 * image, 0, 255).astype(np.uint8)
    return image


def write_text_on_image(images, text):
    """
    :param images: (N, C, H, W) in scale [0, 1]
    :param text: (str, ) * N
    :return: (N, C, H, W) in scale [0, 1]
    """
    if text is None or text[0] is None:
        return images

    images = np.transpose(images, [0, 2, 3, 1])
    images = np.clip(255 * images, 0, 255).astype(np.uint8)

    image_list = []
    for i in range(images.shape[0]):
        img = images[i].copy()
        img = put_multi_line(img, text[i])
        image_list.append(img)
    image_list = np.array(image_list).astype(np.float32)
    image_list = image_list.transpose([0, 3, 1, 2])
    image_list = image_list / 255
    return image_list


def put_multi_line(img, multi_line, h=15):
    for i, line in enumerate(multi_line.split('\n')):
        img = cv2.putText(img, line, (h, h * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
    return img

def square_bbox(bbox, pad=0):
    if not torch.is_tensor(bbox):
        is_numpy = True
        bbox = torch.FloatTensor(bbox)
    else:
        is_numpy = False

    x1y1, x2y2 = bbox[..., :2], bbox[..., 2:]
    center = (x1y1 + x2y2) / 2 
    half_w = torch.max((x2y2 - x1y1) / 2, dim=-1)[0]
    half_w = half_w * (1 + 2 * pad)
    bbox = torch.cat([center - half_w, center + half_w], dim=-1)
    if is_numpy:
        bbox = bbox.cpu().detach().numpy()
    return bbox


def crop_resize(img: np.ndarray, bbox, final_size=224, pad='constant', return_np=True, **kwargs):
    # todo: joint effect
    ndim = img.ndim
    img_y, img_x = img.shape[0:2]

    min_x, min_y, max_x, max_y = np.array(bbox).astype(int)
    # pad
    pad_x1, pad_y1 = max(-min_x, 0), max(-min_y, 0)
    pad_x2, pad_y2 = max(max_x - img_x, 0), max(max_y - img_y, 0)
    pad_dim = ((pad_y1, pad_y2), (pad_x1, pad_x2), )
    if ndim == 3:
        pad_dim += ((0, 0), )
    img = np.pad(img, pad_dim, mode=pad, **kwargs)

    min_x += pad_x1
    max_x += pad_x1
    min_y += pad_y1
    max_y += pad_y1
    
    img = Image.fromarray(img.astype(np.uint8))
    img = img.crop([min_x, min_y, max_x, max_y])
    img = img.resize((final_size, final_size))

    if return_np:
        img = np.array(img)
    return img    