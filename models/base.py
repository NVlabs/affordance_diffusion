# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import importlib
import wandb
import logging
import os
import os.path as osp
import shutil
from omegaconf import OmegaConf
from hydra import main
import torch
import torchvision.utils as vutils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.utilities.distributed import rank_zero_only

from utils.glide_utils import sample
from dataset.dataset import build_dataloader
from utils.logger import LoggerCallback, build_logger
from glide_text2im.text2im_model import Text2ImUNet
from glide_text2im.respace import SpacedDiffusion
from jutils import image_utils


class BaseModule(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.template_size = None
        self.glide_model: Text2ImUNet = None
        self.diffusion: SpacedDiffusion = None
        self.glide_options: dict = {}
        
        self.val_batch = None
        self.train_batch = None
    
    def train_dataloader(self):
        cfg = self.cfg
        dataloader = build_dataloader(cfg, cfg.trainsets, 
            self.glide_model.tokenizer, True, cfg.batch_size, True)
        for data in dataloader:
            self.train_batch = data
            break
        return dataloader

    def val_dataloader(self):
        cfg =self.cfg
        val_dataloader = build_dataloader(cfg,  cfg.testsets,
            self.glide_model.tokenizer, False, cfg.test_batch_size, False)
        for data in val_dataloader:
            self.val_batch = data
            break
        return val_dataloader

    def init_model(self):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [x for x in self.glide_model.parameters() if x.requires_grad],
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.adam_weight_decay,
        )
        return optimizer

    def test_step(self, batch, batch_idx):
        return

    def training_step(self, batch, batch_idx):
        self.train_batch = batch
        loss, losses = self.step(batch, batch_idx)
        if torch.isnan(loss):
            import pdb; pdb.set_trace()


        if self.global_step % self.cfg.print_frequency == 0:
            losses["loss"] = loss
            self.logger.log_metrics(losses, step=self.global_step)
        
            print('[%05d]: %f' % (self.global_step, loss))
            for k, v in losses.items():
                print('\t %08s: %f' % (k, v.item()))
        return loss 
    
    def validation_step(self, batch, batch_idx):
        val_batch = self.val_batch
        train_batch = self.train_batch

        log = {}
        self.vis_input(val_batch, 'val_input/', log)
        self.generate_sample_step(val_batch, 'val/', log)
        shift_batch = shift_image(val_batch)
        self.generate_sample_step(shift_batch, 'val/shift', log, S=1)

        if train_batch is not None:
            self.vis_input(train_batch, 'train_input/', log)
            self.generate_sample_step(train_batch, 'train/', log)
            shift_batch = shift_image(train_batch)
            self.generate_sample_step(shift_batch, 'train/shift', log, S=1)

        self.logger.log_metrics(log, self.global_step)

    def forward(self, batch, **kwargs):
        cfg = self.cfg 
        file_list = []
        samples, sample_list = sample(
            glide_model=self.glide_model,
            glide_options=self.glide_options,
            size=self.template_size,
            batch_size=len(batch[0]),
            guidance_scale=cfg.test_guidance_scale,
            device=self.device,
            prediction_respacing=cfg.sample_respacing,
            image_to_upsample=None,
            val_batch=batch,
            inpaint_enabled=True,
            soft_mask=self.cfg.soft_mask,
            **kwargs,
        )
        return samples, sample_list

    def generate_sample_step(self, batch, pref, log, step=None, S=2):
        cfg = self.cfg 
        if step is None: step = self.global_step
        file_list = []
        step = self.global_step
        for n in range(S):
            samples, sample_list = sample(
                glide_model=self.glide_model,
                glide_options=self.glide_options,
                size=self.template_size,
                batch_size=len(batch[0]),
                guidance_scale=cfg.test_guidance_scale,
                device=self.device,
                prediction_respacing=cfg.sample_respacing,
                image_to_upsample=None,
                val_batch=batch,
                inpaint_enabled=True,
                soft_mask=self.cfg.soft_mask,
            )
            self.vis_samples(batch, samples, sample_list, pref + '%d_' % n, log, step)
        return file_list
    
    def blend(self, mask, bg, r=0.75):
        overlay = mask * (bg * (1-r) + r) + (1 - mask) * bg
        return overlay

    @rank_zero_only
    def vis_samples(self, batch, samples, sample_list, pref, log, step=None):        
        log['%ssample' % pref] = wandb.Image(vutils.make_grid(samples))
        
        fname = osp.join(self.logger.save_dir, '%05d_%s_progress') % (self.global_step, pref)
        image_utils.save_gif(sample_list, fname)
        log['%sprogress' % pref] = wandb.Video(fname + '.gif')
        return 

    @rank_zero_only
    def vis_input(self, batch, pref, log, step=None ):
        log["%sgt" % pref] = wandb.Image(vutils.make_grid(batch[2]))
        overlay = self.blend(batch[4], batch[2])
        log["%smask" % pref] = wandb.Image(vutils.make_grid(overlay))
        log["%sobj" % pref] = wandb.Image(vutils.make_grid(batch[3]))
        return 



def shift_image(batch):
    new_batch = []
    for i, b in enumerate(batch):
        if i != 3:
            new_batch.append(b)
        else:
            new_b = torch.cat([b[1:], b[0:1]], 0)
            new_batch.append(new_b)
    return new_batch


# second stage
@main('../configs', 'train', version_base=None)   
def main_worker(cfg):
    # handle learning rate 
    torch.backends.cudnn.benchmark = True
    module = importlib.import_module(cfg.model.module)
    model_cls = getattr(module, cfg.model.model)
    model = model_cls(cfg, )
    model.init_model()

    if cfg.overwrite:
        logging.warn('#### Dangerous, overwrite %s' % cfg.exp_dir)
        shutil.rmtree(cfg.exp_dir, ignore_errors=True)

    # instantiate model
    if cfg.eval:
        trainer = pl.Trainer(gpus='0,',
                             default_root_dir=cfg.exp_dir,
                             )
        print(cfg.exp_dir, cfg.ckpt)

        model.freeze()
        trainer.test(model=model, verbose=False)
    else:
        os.makedirs(cfg.exp_dir, exist_ok=True)
        with open(osp.join(cfg.exp_dir, 'config.yaml', ), 'w') as fp:
            # OmegaConf.save()
            OmegaConf.save(cfg, fp, True)
        
        logger = build_logger(cfg)

        checkpoint_callback = ModelCheckpoint(
            monitor='step',
            save_top_k=cfg.save_topk,
            mode="max",
            every_n_train_steps=cfg.save_frequency,
            save_last=True,
            dirpath=osp.join(cfg.exp_dir, 'checkpoints'),
            filename='glide-ft-{step}'
        )

        val_kwargs = {}        
        if len(model.train_dataloader()) <cfg.log_frequency:
            val_kwargs['check_val_every_n_epoch'] = int(cfg.log_frequency) // len(model.train_dataloader())
        else:
            val_kwargs['val_check_interval'] = cfg.log_frequency
        model_summary = ModelSummary(2)
        trainer = pl.Trainer(
                             gpus=-1,
                             strategy='ddp',
                             num_sanity_val_steps=cfg.sanity_step,
                             limit_val_batches=1,
                             default_root_dir=cfg.exp_dir,
                             logger=logger,
                             max_steps=cfg.max_steps,
                             callbacks=[model_summary, checkpoint_callback, LoggerCallback()],
                             progress_bar_refresh_rate=None,
                             gradient_clip_val=cfg.model.grad_clip,
                             gradient_clip_algorithm='norm',
                             **val_kwargs,
                             )
        ckpt_path = cfg.get('resume_train_from', None)
        if not osp.exists(ckpt_path):
            ckpt_path = None
        trainer.fit(model, ckpt_path=ckpt_path)
        
    return model

if __name__ == '__main__':
    main_worker()