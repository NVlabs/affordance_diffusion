# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import pickle
from glob import glob
import cv2
import imageio
import numpy as np
import json
import os
import os.path as osp
from hydra import main
import hydra.utils as hydra_utils
import pandas
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_lightning as pl
from models.hand_proxy import build_stn
from utils.train_utils import load_from_checkpoint
from jutils import image_utils
from dataset.dataset import build_imglist_dataloader
from utils import overlay_hand


class CascadeAffordance:
    def __init__(self, what_ckpt, where_ckpt=None, S=2, device='cuda:0', 
        save_dir=None, what=None, where=None, test_name='cascade', cfg=None) -> None:
        self.device = device
        self.S = S
        self.cfg = cfg

        self.what = what
        self.where = where
        self.what_ckpt = what_ckpt
        self.where_ckpt = where_ckpt
        self.where_cfg = None
        self.what_cfg = None
        if what_ckpt is not None:   
            self.save_dir = osp.join(what_ckpt.split('checkpoints')[0],  test_name)
        if where_ckpt is not None and not cfg.save_to_what:
            if where_ckpt.endswith('.pt'):
                self.save_dir = osp.join(osp.dirname(where_ckpt), test_name)
            else:
                self.save_dir = osp.join(where_ckpt.split('checkpoints')[0], test_name)
        if save_dir is not None:
            self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def init_model(self, ):
        where_ckpt = self.where_ckpt
        what_ckpt = self.what_ckpt
        where = self.where
        what = self.what
        device = self.device
        if where_ckpt is not None:
            if where is None:
                where = load_from_checkpoint(where_ckpt)
            self.where = where
            self.where_cfg = where.cfg
            self.where.eval()    
            self.where.to(device)
        if what_ckpt is not None:
            if what is None:
                what = load_from_checkpoint(what_ckpt)
            self.what = what
            self.what_cfg = what.cfg
            self.what.eval()
            self.what.to(device)

        self.splat_to_mask = build_stn(self.what_cfg if self.what_cfg is not None else self.where_cfg).to(device)

        # save config
        with open(osp.join(self.save_dir, 'test_config.json'), 'w') as fp:
            save_dict = {
                'where': where_ckpt, 'what': what_ckpt, 
                'S': self.S, 
                 }
            if self.where is not None:
                save_dict['where_step'] = self.where.global_step
            if self.what is not None:
                save_dict['what_step'] = self.what.global_step
            json.dump(save_dict, fp, indent=2)

    def superres_eval(self, save_dir):
        wsdir = osp.dirname(save_dir)

        cmd = f'cd {hydra_utils.get_original_cwd()}/third_party/EDSR-PyTorch/src;'
        cmd += ' python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results --dir_demo %s --save_dir %s' % \
            (save_dir, osp.join(wsdir, 'tmp'))
        os.system(cmd)
        os.makedirs(osp.join(wsdir, 'superres'), exist_ok=True)
        cmd = 'mv %s %s' % (osp.join(wsdir, 'tmp/results-Demo/*'), osp.join(wsdir, 'superres/'))
        print(cmd)
        os.system(cmd)
        return 

    def save_batch_params(self, params, folder, name, suf=''):
        for n in range(len(params)):
            save_file = osp.join(self.save_dir, folder, name + '_%02d_' % n + suf + '.pkl')
            os.makedirs(osp.dirname(save_file), exist_ok=True)
            with open(save_file, 'wb') as fp:
                pickle.dump(params[n].cpu().detach().numpy(), fp)
        return 

    def save_batch_images(self, images, folder, name, suf=''):
        print(os.getcwd(), osp.join(self.save_dir, folder, name + '_%02d_' % 0 + suf))
        # save images to folder/name_{batch_idx:02d}.png
        for n in range(len(images)):
            image_utils.save_images(images[n:n+1], osp.join(self.save_dir, folder, name + '_%02d_' % n + suf), scale=True)
        return 
    
    def save_batch_images_w_kpts(self, images, kpts, folder, name, suf=''):
        for n in range(len(images)):
            image = image_utils.save_images(images[n:n+1], None, scale=True)
            image = image.astype(np.uint8).copy()
            H = image.shape[0]
            kp = kpts[n].cpu().detach().numpy()
            kp = (kp * 0.5 + 0.5) * H

            image = cv2.circle(image, (int(kp[0]), int(kp[1])), min(H//16, 8), (255, 0, 0), -1)
            fname = osp.join(self.save_dir, folder, name + '_%02d_' % n + suf) + '.png'
            os.makedirs(osp.dirname(fname), exist_ok=True)
            imageio.imwrite(fname, image)
        return 

    def evaluate_epoch_with_fixed(self, val_dataloader):
        # pl.seed_everything(123)
        apply_super =  self.what_cfg is not None and self.what_cfg.side_x == 64
        for s in range(self.S):
            for i, data in enumerate(val_dataloader):
                tokens, masks, reals, inpaint_image, inpaint_mask, mask_param_gt, _ = data

                name = '%04d' % i
                self.save_batch_images_w_kpts(inpaint_image, mask_param_gt[..., 0:2], 'inp', name, 's%d' % s)

                samples, mask_param, inpaint_mask, = self.forward_w_gt(data, )
                self.save_batch_params(mask_param.cpu(), 'mask_param', name, 's%d' % s)
                overlay = image_utils.blend_images(torch.ones_like(inpaint_image), inpaint_image, 1-inpaint_mask)
                self.save_batch_images(overlay, 'mask', name, 's%d' % s)
                self.save_batch_images(samples, 'overall', name, 's%d' % s)
        if apply_super:
            self.superres_eval(osp.join(self.save_dir, 'overall'))
        overlay_hand.batch_main(self.save_dir, save_mask=True)

    def draw_fig(self, val_dataloader):
        pl.seed_everything(123)
        apply_super =  self.what_cfg is not None and self.what_cfg.side_x == 64
        key = 'overall' if not apply_super else 'overallx64'
        for s in range(self.S):
            for i, data in enumerate(val_dataloader):
                tokens, masks, reals, inpaint_image, inpaint_mask, mask_param, _ = data

                name = '%04d' % i
                self.save_batch_images(inpaint_image, 'inp', name)

                samples, mask_param, inpaint_mask, param_list, sample_list = self(data, return_inter=True)

                samples = samples.cpu()
                mask_param = mask_param.cpu()
                inpaint_mask = inpaint_mask.cpu()
                self.save_batch_images(samples, 'overall', name, 's%d' % s)
                image_utils.save_images(inpaint_mask, osp.join(self.save_dir, name+'_mask_s%d' % s), scale=False)
                image_utils.save_images(
                    image_utils.blend_images(torch.ones_like(inpaint_image), inpaint_image, 1-inpaint_mask, r=7), 
                    osp.join(self.save_dir, name+'_mask'), scale=False)
                self.save_batch_images(
                    image_utils.blend_images(torch.ones_like(inpaint_image), inpaint_image, inpaint_mask),
                    # inpaint_mask.cpu() * (0.7+0.3*inpaint_mask) + (1-inpaint_mask.cpu()) * inpaint_image.cpu(), 
                    'mask', name, 's%d' % s)
                self.save_batch_params(mask_param.cpu(), 'mask_param', name, 's%d' % s)
                image_utils.save_gif(torch.stack(sample_list, 0), osp.join(self.save_dir, name + '_process_s%d' % s), scale=True)

                mask_list = []
                blend_list = []
                for param in param_list:
                    m = self.splat_to_mask(param.to(self.device),  H=inpaint_image.shape[-1])
                    b = image_utils.blend_images(torch.ones_like(inpaint_image), inpaint_image, 1-m)
                    mask_list.append(m)
                    blend_list.append(b)
                image_utils.save_gif(torch.stack(mask_list, 0), osp.join(self.save_dir, name + '_mask_s%d' % s), scale=True)
                image_utils.save_gif(torch.stack(blend_list, 0), osp.join(self.save_dir, name + '_blend_s%d' % s), scale=True)
                
                if i >=15:
                    break
        if apply_super:
            self.superres_eval(osp.join(self.save_dir, 'overall'))

    def evaluate_epoch(self, val_dataloader):
        pl.seed_everything(123)
        apply_super =  self.what_cfg is not None and self.what_cfg.side_x == 64
        for s in range(self.S):
            for i, data in enumerate(val_dataloader):
                tokens, masks, reals, inpaint_image, inpaint_mask, mask_param, _ = data

                name = '%04d' % i
                self.save_batch_images(reals, 'gt', name)
                self.save_batch_images(inpaint_image, 'inp', name)

                samples, mask_param, inpaint_mask, = self(data)

                samples = samples.cpu()
                mask_param = mask_param.cpu()
                inpaint_mask = inpaint_mask.cpu()
                self.save_batch_images(samples, 'overall', name, 's%d' % s)
                self.save_batch_images(
                    image_utils.blend_images(torch.ones_like(inpaint_image), inpaint_image, inpaint_mask),
                    'mask', name, 's%d' % s)
                self.save_batch_params(mask_param.cpu(), 'mask_param', name, 's%d' % s)
                if i >=15:
                    break
        if apply_super:
            self.superres_eval(osp.join(self.save_dir, 'overall'))

    def forward_w_gt(self, batch):
        device = self.device
        tokens, masks, reals, inpaint_image, inpaint_mask, mask_param, others = batch
        batch = tokens.to(device), masks.to(device), reals.to(device), \
                inpaint_image.to(device), inpaint_mask.to(device), mask_param.to(device), others
        
        if self.where is not None:
            H = self.where_cfg.side_x
            where_batch = tokens.to(device), masks.to(device), reals.to(device), \
                F.adaptive_avg_pool2d(inpaint_image.to(device), H) , inpaint_mask.to(device), mask_param.to(device), others
            mask_param, param_list = self.where.forward_w_gt(where_batch,  self.cfg.gt_xy, self.cfg.gt_size)
            inpaint_mask = 1 - self.splat_to_mask(
                mask_param, H=inpaint_image.shape[-1], func_ab=lambda x: x**2)
            if self.what_cfg is None or not self.what_cfg.soft_mask:
                inpaint_mask = (inpaint_mask > 0.5).float()
        
        if self.what is not None:
            H = self.what_cfg.side_x
            what_batch = tokens.to(device), masks.to(device), reals.to(device), \
                F.adaptive_avg_pool2d(inpaint_image.to(device), H) , inpaint_mask.to(device), mask_param.to(device), others
            samples, sample_list = self.what(what_batch)
        else:
            samples = inpaint_mask
        return samples, mask_param, inpaint_mask, 

    def __call__(self, batch, use_gt=False, return_inter=False):
        device = self.device
        tokens, masks, reals, inpaint_image, inpaint_mask, mask_param, text = batch
        batch = tokens.to(device), masks.to(device), reals.to(device), \
                inpaint_image.to(device), inpaint_mask.to(device), mask_param.to(device), text

        if self.where is not None and not use_gt:
            H = self.where_cfg.side_x
            mask_param = torch.ones([len(tokens), ] + self.where.template_size, device=device)
            where_batch = tokens.to(device), masks.to(device), reals.to(device), \
                F.adaptive_avg_pool2d(inpaint_image.to(device), H) , inpaint_mask.to(device),  mask_param, text
            mask_param, param_list = self.where(where_batch)
            inpaint_mask = 1 - self.splat_to_mask(
                mask_param, H=inpaint_image.shape[-1], func_ab=lambda x: x**2)
            if self.what_cfg is None or not self.what_cfg.soft_mask:
                inpaint_mask = (inpaint_mask > 0.5).float()
        
        if self.what is not None:
            H = self.what_cfg.side_x
            what_batch = tokens.to(device), masks.to(device), reals.to(device), \
                F.adaptive_avg_pool2d(inpaint_image.to(device), H) , inpaint_mask.to(device), mask_param.to(device), text
            samples, sample_list = self.what(what_batch)
        else:
            out = self.splat_to_mask.forward_image(mask_param, H=inpaint_image.shape[-1])
            samples = image_utils.blend_images(out['image'], inpaint_image, out['mask'], 1) 
            # samples = inpaint_mask
        if not return_inter:
            return samples, mask_param, inpaint_mask
        else:
            return samples, mask_param, inpaint_mask, param_list, sample_list

    def evaluate(self, metric):
        if metric == 'fid': 
            save_dir = self.save_dir
            self.gen_metric(osp.join(save_dir, self.cfg.folder), self.cfg.dirB)
        elif metric == 'srx4': 
            save_dir = self.save_dir
            self.superres_eval(osp.join(save_dir, 'overall'))
        elif metric == 'contact':
            self.contact_metric(osp.join(self.save_dir, self.cfg.folder))
        elif metric == '3d':
            if osp.exists(osp.join(self.save_dir, 'superres')):
                folder = 'superres'
            else:
                folder = 'overall'
            self.three_d_metric(osp.join(self.save_dir, folder), osp.join(self.save_dir, 'recon'))

    def three_d_metric(self, inp_folder, out_folder, overlay=True):
        cmd = 'cd ' + osp.join(hydra_utils.get_original_cwd(), 'third_party/frankmocap') + '; '
        cmd += f'python -m demo.demo_handmocap --input_path {inp_folder} \
            --out_dir {out_folder} \
            --view_type ego_centric --renderer_type pytorch3d --no_display --save_pred_pkl'
        print(cmd)
        os.system(cmd)
        if overlay:
            overlay_hand.batch_main(osp.dirname(out_folder))

    def contact_metric(self, folder):
        import sys
        sys.path.insert(0, f'{hydra_utils.get_original_cwd()}/third_party/frankmocap/')
        from handmocap.hand_bbox_detector import Ego_Centric_HOI_Detector        
        self.hoi_detector = Ego_Centric_HOI_Detector()

        img_list = sorted(glob(osp.join(folder, "*.png")))
        valid = []
        confidence = []
        for img_file in tqdm(img_list):
            hand_list, contact_list = self.hoi_detector(img_file)

            if len(contact_list) > 0:
                # TODO？
                valid.append(contact_list[0]['state'])
                confidence.append(contact_list[0]['score'])
                if len(contact_list) > 1:
                    print(osp.basename(img_file), contact_list, len(contact_list))
            else:
                valid.append(-1)
                confidence.append(-1)
        valid = np.array(valid)
        score = np.mean(valid==3) * 100
        for num in range(-1, 5):
            print(num, ':', np.mean(valid==num) * 100)
        print('contact hand recall', score)
        with open(osp.join(self.save_dir, 'num_contact_recall.txt'), 'w') as fp:
            fp.write('contact hand recall: %.2f perc \n' % score)
            for num in range(-1, 5):
                fp.write(f'contact Type {num} recall: %.2f perc \n' % (np.mean(valid==num) * 100))
        df = pandas.DataFrame({
            'name': [osp.basename(e) for e in img_list],
            'score': confidence,
            'state': valid,
            })
        df.to_csv(osp.join(self.save_dir, 'contact.csv'))

    def gen_metric(self, folder_pred, folder_gt):
        from fid_score.fid_score import FidScore
        print('folder A B', len(glob(osp.join(folder_pred, '*.png'))), len(glob(osp.join(folder_gt, '*.png'))))
        fid = FidScore([folder_pred, folder_gt], self.device, 32)
        score = fid.calculate_fid_score()        
        print('FID: ', score)
        with open(osp.join(self.save_dir, 'num_fid.txt'), 'w') as fp:
            fp.write('fid: %f\n' % score)
        return 


@torch.no_grad()
@main('configs', 'test', version_base=None)   
def main_worker(cfg):
    wrapper = CascadeAffordance(cfg.what_ckpt, cfg.where_ckpt, 
        test_name=cfg.test_name, S=cfg.test_num, cfg=cfg, save_dir=cfg.dir)
    cfg.dir = wrapper.save_dir
    cfg.use_flip = False
    if not cfg.dry:
        wrapper.init_model()
        if wrapper.what_cfg is not None:
            cfg.side_x = wrapper.what_cfg.side_x
        if wrapper.where_cfg is not None:
            cfg.side_x = max(cfg.side_x, wrapper.where_cfg.side_x)
        cfg.side_x = 256
        val_dataloader = build_imglist_dataloader(cfg, cfg.data.data_dir, 
            cfg.data.split, False, cfg.batch_size, False)
        
        if cfg.mode == 'default':
            wrapper.evaluate_epoch(val_dataloader)
        elif cfg.mode == 'hijack':
            wrapper.evaluate_epoch_with_fixed(val_dataloader)
        else:
            raise NotImplementedError(cfg.mode)

    for metric in cfg.metric.split('+'):
        wrapper.evaluate(metric)


if __name__ == "__main__":
    main_worker()