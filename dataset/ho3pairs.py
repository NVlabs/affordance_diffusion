# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import numpy as np
import os.path as osp
from random import randint, uniform
import cv2
import PIL
from PIL import Image, ImageFilter
import pandas as pd
import json
import torch as th
import torchvision.transforms.functional as F
from torchvision import transforms as T
from torchvision.transforms import Compose, RandomApply, ToPILImage, ToTensor
from torch.utils.data import Dataset
from utils.glide_utils import get_uncond_tokens_mask
from utils.train_utils import pil_image_to_norm_tensor


def random_resized_crop(image, shape, resize_ratio=1.0, return_T=False):
    """
    Randomly resize and crop an image to a given size.

    Args:
        image (PIL.Image): The image to be resized and cropped.
        shape (tuple): The desired output shape.
        resize_ratio (float): The ratio to resize the image.
    """
    image_transform = T.RandomResizedCrop(shape, scale=(resize_ratio, 1.0), ratio=(1.0, 1.0))
    if not return_T:
        return image_transform(image)
    return image_transform


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

        
class HO3Pairs(Dataset):
    def __init__(
        self,
        folder="/ws-judyye/output/inpaint/100doh/",
        split='train_handcrop.csv',
        side_x=64,
        side_y=64,
        resize_ratio=0.75,
        shuffle=False,
        tokenizer=None,
        mask_mode='lollipop',
        use_flip=False,
        is_train=True,
        cfg={},
        data_cfg={},
    ):
        super().__init__()
        self.data_cfg = data_cfg
        self.cfg = cfg
        self.data_dir = folder
        self.split = split
        self.image_dir = osp.join(folder, 'glide_hoi/{}.png')
        self.obj_dir = osp.join(folder, 'glide_obj/{}.png')
        self.sub_dir = osp.join(folder, '%s/{}.png' % cfg.sub_dir)
        self.mask_dir = osp.join(folder, 'det_mask/{}.png')
        self.hand_dir = osp.join(folder, 'det_hand/{}.json')
        self.hoi_box_dir = osp.join(folder, 'hoi_box/{}.json')

        self.mask_mode = mask_mode
        self.use_flip = use_flip
        self.is_train = is_train

        self.transform = Compose(
            [RandomApply([GaussianBlur([.1, 2.])], p=cfg.jitter_p)]
        )
        # self.image_files = list(iou_dict.keys())
        self.image_files = []
        if '.csv' in split:
            df = pd.read_csv(split)
            for i, data in df.iterrows():
                self.image_files.append(
                    '{}_frame{:04d}'.format(
                        data['vid_index'].replace('/', '_'), data['frame_number']))
        else:
            self.image_files = [index.strip() for index in open(split)]

        self.resize_ratio = resize_ratio

        self.shuffle = shuffle
        self.prefix = folder
        self.side_x = side_x
        self.side_y = side_y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_files)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
    
    def load_hand_box(self, ind):
        obj = json.load(open(self.hand_dir.format(self.image_files[ind])))
        box_list = []
        def xywh2xyxy(xywh):
            x1, y1, w, h = xywh
            return [x1, y1, x1+w, y1+h]

        if self.data_cfg.get('xywh', True):
            f_box = xywh2xyxy
        else:
            f_box = lambda x: x
        # to fix hand box file bug
        if isinstance(obj['hand_bbox_list'][0], list):
            obj['hand_bbox_list'] = obj['hand_bbox_list'][0]
        for hand_box in obj['hand_bbox_list']:
            if 'right_hand' in hand_box:
                box_list.append(f_box(hand_box['right_hand']))
            if 'left_hand' in hand_box:
                box_list.append(f_box(hand_box['left_hand']))
        return np.array(box_list)

    def __getitem__(self, ind):
        image_file = self.image_dir.format(self.image_files[ind])

        # null text
        tokens, mask = get_uncond_tokens_mask(self.tokenizer)
        text = ''

        try:
            original_pil_image = PIL.Image.open(image_file).convert("RGB")
        except (OSError, ValueError) as e:
            print(f"An exception occurred trying to load file {image_file}.", self.split)
            print(f"Skipping index {ind}")
            print(e)
            return self.skip_sample(ind)

        # get corresponding object-only image
        try:
            obj_file = self.get_obj_file(ind)
            mask_file = self.mask_dir.format(self.image_files[ind])

            original_pil_obj = PIL.Image.open(obj_file).convert("RGB")
            original_pil_mask = PIL.Image.open(mask_file).convert("RGB")
            original_pil_obj = self.preprocess_obj(original_pil_obj)
        except (FileNotFoundError, OSError, ValueError, cv2.error) as e:
            print(f"An exception occurred trying to load file {obj_file, mask_file}.", self.split)
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        trans = random_resized_crop(original_pil_image, (self.side_x, self.side_y), resize_ratio=self.resize_ratio, return_T=True)
        i, j, h, w = trans.get_params(original_pil_image, trans.scale, trans.ratio)
        base_pil_image = F.resized_crop(original_pil_image, i, j, h, w, trans.size, trans.interpolation)
        base_pil_obj = F.resized_crop(original_pil_obj, i, j, h, w, trans.size, trans.interpolation)
        base_pil_mask = F.resized_crop(original_pil_mask, i, j, h, w, trans.size, trans.interpolation)
        if self.mask_mode == 'lollipop':
            try:
                fHand_box = self.load_hand_box(ind)
                hoi_box = json.load(open(self.hoi_box_dir.format(self.image_files[ind])))
            except FileNotFoundError as e:
                print('no bbox, ', self.hoi_box_dir.format(self.image_files[ind]))
                print(e)
                return self.skip_sample(ind) 
            bboxes = self.transform_box_to_inp(fHand_box, hoi_box, (j, i, j+w, i+h), trans.size)
        else:
            bboxes = None

        if uniform() > 0.5 and self.use_flip:
            base_pil_image = F.hflip(base_pil_image)
            base_pil_obj = F.hflip(base_pil_obj)
            base_pil_mask = F.hflip(base_pil_mask)
            if bboxes is not None: 
                bboxes[..., 0] = self.side_x - bboxes[..., 0]
                bboxes[..., 2] = self.side_x - bboxes[..., 2]

        try:
            base_pil_mask, mask_param = self.preprocess_mask(
                base_pil_mask, self.is_train, bboxes=bboxes, **self.cfg.jitter, iou_th=self.data_cfg.get('iou', 0.5), ind=ind)
        except cv2.error as e:
            print(f"An exception occurred preprocessing mask {mask_file}.", self.split)
            print(e)
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        base_tensor = pil_image_to_norm_tensor(base_pil_image)
        base_obj = pil_image_to_norm_tensor(base_pil_obj)
        base_mask = (th.FloatTensor(np.asarray(base_pil_mask)) > 0)[None]
        return th.LongTensor(tokens), th.BoolTensor(mask), base_tensor, \
            base_obj, base_mask, mask_param.astype(np.float32).reshape(-1), text

    def preprocess_obj(self, image):
        if not self.is_train:
            return image
        # blur
        image = self.transform(image)
        if uniform() < self.cfg.jitter_p:  # w prob p
            image = ToTensor()(image)
            image  += 0.05 * th.randn_like(image)
            image = ToPILImage()(image.clamp(0, 1))
        return image

    def get_obj_file(self, ind):
        obj_file = self.obj_dir.format(self.image_files[ind])
        if self.is_train and uniform() < self.cfg.sub_p:  # subsitute with prob p
            sub_file = self.sub_dir.format(self.image_files[ind])
            if osp.exists(sub_file):
                obj_file = sub_file
        return obj_file

    def transform_box_to_inp(self, fBox, hoi_box, crop_box, size):
        """
        fBox: box to transform, originally in frame coord
        hoi_box: original box
        """
        fBox = np.array(fBox)
        if len(fBox) == 0:
            return None
        hoi_box = np.array(hoi_box)[None]
        crop_box = np.array(crop_box)[None]

        fWh = fBox[..., 2:4] - fBox[..., 0:2]
        oldsize = crop_box[..., 2:4] - crop_box[..., 0:2]
        hX1y1 = fBox[..., 0:2] - hoi_box[..., 0:2]
        cX1y1 = (hX1y1 - crop_box[..., 0:2]) * size / oldsize
        cWh = fWh * size / oldsize
        return np.concatenate([cX1y1, cX1y1 + cWh], -1)

    def preprocess_mask(self, mask, is_train, bboxes=None, **kwargs):

        mask = np.array(mask)
        if np.array(mask).ndim == 3:
            mask = np.array(mask)[..., 0]

        if self.mask_mode == 'gt':
            mask_param = np.zeros([6])
        elif self.mask_mode == 'lollipop':
            mask, mask_param = lollipop(mask, is_train, bboxes, **kwargs, one_hand=self.cfg.one_hand)
        elif self.mask_mode == 'pose':
            mask, mask_param = self.pose(kwargs.get('ind'))
        else:
            raise NotImplementedError

        mask = Image.fromarray(mask)
        return mask, mask_param


def associate_mask_box(boxes, masks, one_hand):
    """

    :param boxes: (4), use the first boxes
    :param masks: (H, W) non hand mask --> found the mask region by connnected component
    :param one_hand: true: --> found the mask region by connnected component / false
    :raises cv2.error: _description_
    : reutrn:  one-hand mask? (0, 255)
    """
    binary_map = (masks < 122.5).astype(np.uint8) * 255

    if not one_hand:
        return binary_map
    
    H = binary_map.shape[0]
    k = H*4//64
    bc = (boxes[0:2] + boxes[2:4]) / 2
    erode = cv2.erode(erode, np.ones((k, k), np.uint8), iterations=1)  # dilate vs iteration?
    erode = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for contour in contours:
        d = cv2.pointPolygonTest(contour, (int(bc[0]), int(bc[1])), True,)
        dist.append(d)  # inside: +max 
    dist = np.array(dist)
    ind = dist.argmax()
    contour = contours[ind]

    img = np.zeros((H, H), np.uint8)
    cv2.fillPoly(img, pts =[contour], color=(255,255,255))
    img = cv2.dilate(img, np.ones((k+3, k+3), np.uint8))  
    return img


def lollipop(mask, jitter, boxes, xy=0, ab=0, theta=0, scale=1., one_hand=False, **kwargs):
    iou_th = kwargs.get('iou_th', 0.5)
    # if associate with ??
    # if kwargs.get('test_iou', False)
    if boxes is None:
        raise cv2.error
    hand_box = boxes[0]
    mask = associate_mask_box(hand_box, mask, one_hand) 
    jxy = xy
    bc = (hand_box[0:2] + hand_box[2:4]) / 2
    
    Y, X = np.where(mask > 0)
    if len(Y) < 50:
        raise cv2.error('Too few points %d' % len(Y))
    XY = np.stack([X, Y], -1)
    xy = np.mean(XY - bc, axis=0)
    angle = np.arctan2(xy[1], xy[0])  # y, x

    x1, y1, x2, y2 = hand_box.tolist()
    x, y = [(x1 + x2) / 2, (y1 + y2) / 2]
    size = max((x2 - x1) / 2, (y2 - y1) / 2)
    
    if jitter:
        rand = lambda x: (uniform() * 2 * x - x)  # [-x, x]
        # rand = lambda x: (0 * 2 * x - x)  # [-x, x]
        # print('TODO change back ', x)
        x += x *  rand(jxy)
        y += y * rand(jxy)
        size += size * rand(ab)
        angle += angle * rand(theta/180*np.pi)

    norm = xy = np.array([np.cos(angle), np.sin(angle)])
    length = 2 * mask.shape[0]
    overal_canvas = None
    x_end, y_end = x +  length * norm[0], y + length * norm[1]
    canvas = np.zeros_like(mask)
    cv2.circle(canvas, (int(x), int(y)), int(size), color=(255, 255, 255), thickness=-1)
    cv2.line(canvas, (int(x), int(y)), (int(x_end), int(y_end)), color=(255, 255, 255), thickness=int(size))
    canvas = (canvas < 122.5).astype(np.uint8) * 255
    def iou_fn(a, b):
        mask1 = a > 0
        mask2 = b > 0
        iou = np.logical_and(mask1, mask2).sum() / np.logical_or(mask1, mask2).sum()
        return iou
    iou = iou_fn(255 - canvas, mask)
    if iou < iou_th:
        raise cv2.error('%f' % iou)
    overal_canvas = canvas
    canvas = overal_canvas

    # convert param
    H, W = mask.shape[0:2]
    norm = norm / np.linalg.norm(norm)
    # todo: size! size=1
    param = np.array([(x / W) * 2 - 1, (y / H)*2-1, (size/W*4)**0.5, norm[0], norm[1]])

    # canvas = np.transpose(canvas, [2, 0, 1])
    return canvas, param


def convert(ellipse_list, origX, origY):
    """

    :param ellipse_list: _description_
    :param origX: _description_
    :param origY: _description_
    :return: (K, 6?)
    """
    exp_para = np.array(ellipse_list)  # (K, 5)
    x = exp_para[..., 0] / origX
    y = exp_para[..., 1] / origY
    x = 2 * x - 1
    y = 2 * y - 1

    a, b = (exp_para[..., 2] + 0.5) / origX, (exp_para[..., 3] + 0.5) / origY 

    angle = exp_para[..., 4]
    
    covs = np.stack([
        a, b,
        np.cos(angle / 180 * np.pi), 
        np.sin(angle / 180 * np.pi)
    ], -1)
    
    params = np.concatenate([x[..., None], y[..., None], covs], -1)
    return params

