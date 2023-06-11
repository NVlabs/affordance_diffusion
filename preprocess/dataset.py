# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

import json
import os
import os.path as osp
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import imageio
import numpy as np
import pandas

from torch.utils.data import Dataset
import image_utils


class HOI4D(Dataset):
    def __init__(self, data_dir, save_dir, split, save_index='HOI4D_glide') -> None:
        super().__init__()
        # self.index_list = pandas.read_csv(osp.join(data_dir, 'Sets', split + '.csv'))
        self.index_list = pandas.read_csv(split)

        data_dir = osp.join(data_dir)
        save_dir = osp.join(save_dir)
        
        self.image_dir = osp.join(data_dir, 'HOI4D_release/{}/align_frames/{:04d}.png')
        self.mask_dir = osp.join(data_dir, 'HOI4D_annotations/{}/2Dseg/shift_mask/{:05d}.png')
        
        self.save_hoi = osp.join(save_dir, save_index, 'glide_hoi/{}_frame{:04d}.png')
        os.makedirs(osp.dirname(self.save_hoi), exist_ok=True)
        self.save_obj = osp.join(save_dir, save_index, 'glide_obj/{}_frame{:04d}.png')
        os.makedirs(osp.dirname(self.save_obj), exist_ok=True)
        self.save_mask = osp.join(save_dir, save_index, 'det_mask/{}_frame{:04d}.png')
        os.makedirs(osp.dirname(self.save_mask), exist_ok=True)
        self.save_box = osp.join(save_dir, save_index, 'hoi_box/{}_frame{:04d}.json')
        os.makedirs(osp.dirname(self.save_box), exist_ok=True)
        
        self.error = osp.join(save_dir, save_index, 'errors/{}.txt')
        os.makedirs(osp.dirname(self.error), exist_ok=True)

    def __len__(self):
        return len(self.index_list)
    
    def get_bbox(self, mask, H, W):
        bbox = image_utils.mask_to_bbox((mask[..., 0] + mask[..., 2]) > mask.max() / 2)
        bbox = image_utils.square_bbox(bbox, pad=0.8)
        bbox = image_utils.intersect_box(bbox, np.array([0,0,W-1,H-1]))
        bbox = image_utils.square_bbox_no_black(bbox, Ymax=H, Xmax=W,)
        hoi_box = bbox
        return hoi_box
    
    def get_save_index(self, ind):
        data = self.index_list.iloc[ind]
        save_index = (data['vid_index'].replace('/', '_'), data['frame_number'])
        return save_index

    def __getitem__(self, ind):
        data = self.index_list.iloc[ind]
        save_index = self.get_save_index(ind)
        frame_num = data['frame_number']
        index = (data['vid_index'], data['frame_number'])
        try:
            hoi_image = Image.open(self.image_dir.format(data['vid_index'], frame_num + 1))  
        except (PIL.UnidentifiedImageError, FileNotFoundError) as e:
            print('reextract', index)
            print(e)
            os.system('touch %s' % self.error.format(data['vid_index'].replace('/', '_')))
            return None

        if osp.exists(self.mask_dir.format(*index)):
            mask = imageio.imread(self.mask_dir.format(*index))
        elif osp.exists(self.mask_dir.format(*index).replace('shift_mask', 'mask')):
            mask = imageio.imread(self.mask_dir.format(*index).replace('shift_mask', 'mask'))
        else: 
            return None
        mask = np.array(mask)
        if mask.ndim == 2:
            print(index, self.mask_dir.format(*index), mask.shape)
            mask = np.stack([mask] * 3, -1)
        W, H = hoi_image.size
        hoi_box = self.get_bbox(mask, H, W)

        try:
            hoi_image = np.array(hoi_image.crop(hoi_box))
        except SystemError:
            print(hoi_box, W, H, index)
            return None
        not_hand_mask = Image.fromarray(
            np.logical_or(
                np.logical_or(
                    mask[..., 1] < mask.max() / 2, 
                    mask[..., 0] > mask.max() / 2, 
            ), mask[..., 2] > mask.max() / 2, 
            ) ).crop(hoi_box)
        mask = np.array(not_hand_mask).astype(np.uint8) * 255

        inp_file = self.save_box.format(*save_index)
        if not osp.exists(inp_file): json.dump(hoi_box.tolist(), open(inp_file, 'w'))

        inp_file = self.save_hoi.format(*save_index)
        if not osp.exists(inp_file): imageio.imwrite(inp_file, hoi_image)

        inp_file = self.save_mask.format(*save_index)
        if not osp.exists(inp_file): imageio.imwrite(inp_file, mask)

        min_obj, min_len = '' ,1000000
        for obj in data['obj'].split('/'):
            if len(obj) < min_len:
                min_len = len(obj)
                min_obj = obj
        if min_obj == 'Watercup':
            min_obj = 'mug'

        out = {
            'inp_file': self.save_hoi.format(*save_index),
            'out_file': self.save_obj.format(*save_index),
            'mask_file': self.save_mask.format(*save_index),
            
            'prompt': min_obj,
        }

        return out
