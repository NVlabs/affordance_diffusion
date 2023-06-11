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

import numpy as np
import torch

def mask_to_bbox(mask, mode='minmax', rate=1):
    """
    Args:
        mask (H, W)
    """
    h_idx, w_idx = np.where(mask > 0)
    if len(h_idx) == 0:
        return np.array([0, 0, 0, 0])

    if mode == 'minmax':
        y1, y2 = h_idx.min(), h_idx.max()
        x1, x2 = w_idx.min(), w_idx.max()
    elif mode == 'com':
        y_c, x_c = h_idx.mean(), w_idx.mean()
        y_l, x_l = h_idx.std(), w_idx.std()

        x1, y1 = x_c - x_l, y_c - y_l
        x2, y2 = x_c + x_l, y_c + y_l
    elif mode == 'med':
        h_idx, w_idx = np.sort(h_idx), np.sort(w_idx)

        idx25 = len(h_idx) // 4
        idx75 = len(h_idx) * 3 // 4
        y_c, x_c = h_idx[len(h_idx) // 2], w_idx[len(w_idx) // 2]
        y_l, x_l = h_idx[idx75] - h_idx[idx25], w_idx[idx75] - w_idx[idx25]

        x1, y1 = x_c - rate*x_l, y_c - rate*y_l
        x2, y2 = x_c + rate*x_l, y_c + rate*y_l

    return np.array([x1, y1, x2, y2])



def square_bbox_no_black(bbox, Ymax, Xmax, pad=0):
    bbox = square_bbox(bbox, pad)
    x1, y1, x2, y2 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    if x1 < 0:
        x1 = 0
        x2 = min(x2 + 0 - x1, Xmax - 1)
    if x2 >= Xmax-1:
        x2 = min(Xmax - 1, x2)
        x1 = max(0, x1 - (Xmax - 1 - x2))
    if y1 < 0:
        y1 = 0
        y2 = min(y2 + 0 - y1, Ymax - 1)
    if y2 >= Ymax-1:
        y2 = Ymax - 1
        y1 = max(0, y1 - (Ymax - 1 - y2))
    # intersect 
    center = np.stack([(x1 + x2) / 2, (y1 + y2) / 2], -1)
    size = np.array(min((x2 - x1) / 2, (y2 - y1) / 2))[..., None]
    
    bbox = np.concatenate([center - size, center + size], -1)
    return bbox


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



def intersect_box(*bboxes):
    bboxes = np.array(bboxes)
    x1 = bboxes[:, 0].max()
    y1 = bboxes[:, 1].max()
    x2 = bboxes[:, 2].min()
    y2 = bboxes[:, 3].min()
    return np.array([x1, y1, x2, y2])