# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import pickle
import numpy as np
import os.path as osp
from random import randint
import PIL
from PIL import Image
import torch as th
import torchvision.transforms.functional as F
from torchvision.transforms import Resize, CenterCrop, Compose
from torch.utils.data import Dataset


from glide_text2im.tokenizer.bpe import get_encoder
from utils.glide_utils import get_uncond_tokens_mask
from utils.train_utils import pil_image_to_norm_tensor
from glob import glob
from jutils import image_utils


class ObjImgDataset(Dataset):
    def __init__(
            self,
            folder="",
            side_x=64,
            side_y=64,
            shuffle=False,
            tokenizer=None,
            use_flip=False,
            is_train=True,
            split=None,
            cfg={},
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if split is not None:
            self.image_files = [folder.replace('*',  e.strip()) for e in open(split)]
        else:
            print('glob ', folder)
            self.image_files = sorted(glob(folder))

        self.use_flip = use_flip
        self.is_train = is_train

        self.shuffle = shuffle
        self.side_x = side_x
        self.side_y = side_y
        if tokenizer is None:
            tokenizer = get_encoder()
        self.tokenizer = tokenizer
        
        self.crop_transform = Compose([
            Resize(side_x),
            CenterCrop(side_x),
        ])

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

    def __getitem__(self, ind):
        image_file = self.image_files[ind]
        tokens, mask = get_uncond_tokens_mask(self.tokenizer)

        try:
            original_pil_image = PIL.Image.open(image_file).convert("RGB")
        except (OSError, ValueError) as e:
            print(f"An exception occurred trying to load file {image_file}.", self.split)
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        W, H = original_pil_image.size
        # try load gt mask_param
        mask_param, gt_mask = self.load_hijack_param(ind, H, W)

        if self.cfg.pad is not None:
            W, H = original_pil_image.size
            box = image_utils.square_bbox([0, 0, W, H])  # X, Y
            original_pil_image = image_utils.crop_resize(np.array(original_pil_image), box, self.side_x, pad=self.cfg.pad)
            base_pil_image = Image.fromarray(original_pil_image)
        else:
            base_pil_image = self.crop_transform(original_pil_image)
            gt_mask = self.crop_transform(gt_mask)
        
        if np.random.rand() > 0.5 and self.use_flip:
            base_pil_image = F.hflip(base_pil_image)
            gt_mask = F.hflip(gt_mask)
            mask_param[0:2] = -mask_param[0:2]
            
        base_tensor = pil_image_to_norm_tensor(base_pil_image)
        gt_mask = (th.FloatTensor(np.asarray(gt_mask)))[None]

        H, W = self.side_y, self.side_x
        return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor, \
            base_tensor, gt_mask, mask_param, ''
    
    def load_hijack_param(self, ind, H, W):
        mask_inp = 5 
        mask_param = np.zeros([mask_inp]).astype(np.float32)
        heatmap = np.zeros([H, W]).astype(np.uint8)
        mask_param[2:] = 1
        anno_file =  self.image_files[ind] + '.pkl'
        if osp.exists(anno_file):
            info = pickle.load(open(anno_file, 'rb'))
            if 'kpts' in info:
                kpts_list = info['kpts']
                inds = randint(0, len(kpts_list)-1)
                print(ind, inds)
                mask_param[0:2] = kpts_list[inds]
            if 'size' in info:
                mask_param[2] = info['size']
            if 'heatmap' in info:
                heatmap = info['heatmap']
                heatmap = (255 * heatmap / (heatmap.max() + 1e-12)).astype(np.uint8)
        heatmap = Image.fromarray(heatmap)
        return mask_param, heatmap

