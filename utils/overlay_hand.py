# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import argparse
from glob import glob
from PIL import Image
from torchvision.transforms import ToTensor
import os
import os.path as osp
import numpy as np
import torch
import pickle
from jutils import hand_utils, image_utils, geom_utils, mesh_utils
from pytorch3d.renderer import PerspectiveCameras
from tqdm import tqdm

device = 'cuda:0'
bad_cnt = 0

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', )
    return parser.parse_args()


def batch_main(args_dir, **kwargs):
    wrapper = hand_utils.ManopthWrapper().to(device)
    pred_list = sorted(glob(osp.join(args_dir, 'recon/mocap/*_prediction_result.pkl')))
    print(osp.join(args_dir, 'recon/mocap/*_prediction_result.pkl'), len(pred_list))
    save_dir = osp.join(args_dir, 'recon/overlay')
    os.makedirs(save_dir, exist_ok=True)
    cnt = 0
    for pred_file in tqdm(pred_list):
        cnt += 1
        index = osp.basename(pred_file).split('_prediction_result')[0]
        obj_file = osp.join(args_dir, 'inp', index[:-2] + '.png')

        overlay_one(obj_file, pred_file, osp.join(save_dir, index), wrapper, **kwargs)
    print(bad_cnt)

    
def overlay_one(image_file, fname,  save_file, wrapper, save_mask=False):
    global bad_cnt

    a = pickle.load(open(fname, 'rb'))
    try:
        inp = (ToTensor()(Image.open(image_file)) * 2 - 1)[None]
    except FileNotFoundError:
        print(image_file)
        return
    hand = a['pred_output_list'][0]
    for hand_type, hand_info in hand.items():
        if len(hand_info) > 0:
            break
    # left_hand = a['pred_output_list'][0]['left_hand']
    if 'pred_hand_pose' in hand_info:
        # cv2.imwrite(osp.join(save_dir, 'crop.png'), hand['img_cropped'])
        data = process_mocap_predictions(hand_info, 256, 256, wrapper, hand_type)
        cHand, _ = wrapper(data['cTh'].to(device), data['hA'].to(device))
        cHand.textures = mesh_utils.pad_texture(cHand, 'blue')
        cameras = PerspectiveCameras(data['cam_f'],data['cam_p'], device=device)
        iHand = mesh_utils.render_mesh(cHand, cameras, out_size=256)

        if hand_type == 'left_hand':
            iHand['image'] = torch.flip(iHand['image'], [-1])
            iHand['mask'] = torch.flip(iHand['mask'], [-1])
        out = image_utils.blend_images(iHand['image'] * 2 - 1, inp, iHand['mask'], 1)
        if save_mask:
            image_utils.save_images(iHand['mask'], save_file + '_mask', scale=True)

            iHand = mesh_utils.render_mesh(cHand, cameras, out_size=512)
            if hand_type == 'left_hand':
                iHand['image'] = torch.flip(iHand['image'], [-1])
                iHand['mask'] = torch.flip(iHand['mask'], [-1])
            image_utils.save_images(iHand['mask'], save_file + '_maskx512', scale=True)
            image_utils.save_images(iHand['image']* 2 - 1, save_file + '_handx512', scale=True)
    else:
        bad_cnt += 1
        print('no hand!!!')
        out = inp
    image_utils.save_images(out, save_file, scale=True)



def get_camera(pred_cam, hand_bbox_tl, bbox_scale, bbox, hand_wrapper, hA, rot, fx=10):
    device = hA.device
    new_center = (bbox[0:2] + bbox[2:4]) / 2
    new_size = max(bbox[2:4] - bbox[0:2])
    cam, topleft, scale = image_utils.crop_weak_cam(
        pred_cam, hand_bbox_tl, bbox_scale, new_center, new_size)
    s, tx, ty = cam
    
    f = torch.FloatTensor([[fx, fx]]).to(device)
    p = torch.FloatTensor([[0, 0]]).to(device)

    translate = torch.FloatTensor([[tx, ty, fx/s]]).to(device)
    
    _, joints = hand_wrapper(
        geom_utils.matrix_to_se3(geom_utils.axis_angle_t_to_matrix(rot)), 
        hA)
    
    cTh = geom_utils.axis_angle_t_to_matrix(
        rot, translate - joints[:, 5])
    return cTh, f, p


# from https://github.com/JudyYe/ihoi
def process_mocap_predictions(one_hand, H, W, hand_wrapper=None, hand_side='right_hand'):
    if hand_side == 'left_hand':
        one_hand['pred_camera'][..., 1] *= -1
        one_hand['pred_hand_pose'][:, 1::3] *= -1
        one_hand['pred_hand_pose'][:, 2::3] *= -1
        old_size = 224 / one_hand['bbox_scale_ratio']
        one_hand['bbox_top_left'][..., 0] = W - (one_hand['bbox_top_left'][..., 0] + old_size)  # xy

    pose = torch.FloatTensor(one_hand['pred_hand_pose']).to(device)
    rot, hA = pose[..., :3], pose[..., 3:]
    hA = hA + hand_wrapper.hand_mean

    hoi_bbox = np.array([0, 0, W, H])
    
    cTh, cam_f, cam_p = get_camera(one_hand['pred_camera'], one_hand['bbox_top_left'], one_hand['bbox_scale_ratio'], hoi_bbox, hand_wrapper, hA, rot)

    data = {
        'cTh': geom_utils.matrix_to_se3(cTh).to(device),
        'hA': hA.to(device),
        'cam_f': cam_f.to(device),
        'cam_p': cam_p.to(device)
    }
    return data


if __name__ == '__main__':
    image_file = 'output/release/layout/hoi4d/recon/recon/rendered/0000_00_s0.jpg'
    fname = 'output/release/layout/hoi4d/recon/mocap/0000_00_s0_prediction_result.pkl'
    save_dir = 'output/vis_hand/'
    os.makedirs(save_dir, exist_ok=True)

    args = parser_args()
    batch_main(args.dir)