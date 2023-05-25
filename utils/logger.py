# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
import logging
import os
import os.path as osp
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union
from pytorch_lightning import Callback
import wandb
import torchvision.utils as vutils
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.logger import _add_prefix
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class LoggerCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_batch_start(self, trainer, pl_module) -> None:
        if hasattr(pl_module.logger, 'commit'):
            pl_module.logger.commit(step=pl_module.global_step)
        else:
            logging.info('no commit avaialbe')

class LFSLogger(LightningLoggerBase):
    """
    local system logger
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: Union[str, bool] = False,
        experiment=None,
        prefix: Optional[str] = "",
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        agg_default_func: Optional[Callable[[Sequence[float]], float]] = None,
        **kwargs,
    ):
        super().__init__(agg_key_funcs=agg_key_funcs, agg_default_func=agg_default_func)
        self._save_dir = save_dir + '/log'
        # os.system('rm -r %s ' % self._save_dir)
        os.makedirs(self._save_dir, exist_ok=True)
        self._name = name
        self._version = version

    @property
    def save_dir(self) -> Optional[str]:
        return self._save_dir
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> Union[int, str]:
        return self._version
    
    @rank_zero_only    
    def log_hyperparams(self, params, *args, **kwargs):
        # params = _convert_params(params)
        # params = _flatten_dict(params)
        # params = _sanitize_callable_params(params)
        # self.experiment.config.update(params, allow_val_change=True)
        return 
        
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        print('Global Step [%05d] ' % step)
        for k,v in metrics.items():
            print(k, v)
            if isinstance(v, wandb.Image):
                fname = os.path.join(self.save_dir, '%d_%s.jpg' % (step, k.replace('/', '_')))
                os.makedirs(osp.dirname(fname), exist_ok=True)
                v.image.save(fname)
    
    @rank_zero_only
    def log_image(self, key: str, images: List[Any], step: Optional[int] = None, **kwargs: str) -> None:
        caption = kwargs.get('caption', [str(e) for e in range(len(images))])
        for i, image in enumerate(images):
            fname = os.path.join(self.save_dir, '%d_%s_%s.jpg' % (step, key.replace('/', '_'), caption))
            vutils.save_image(image / 2 + 0.5, fname)
            print('save to ', fname)
        return 

    def watch(self, *args, **kwargs):
        return 

    def unwatch(self, *args, **kwargs):
        return 

    @rank_zero_only
    def commit(self, step=None):
        return
        
class MyWandbLogger(WandbLogger):
    def __init__(self, name: Optional[str] = None, save_dir: Optional[str] = None, offline: Optional[bool] = False, id: Optional[str] = None, anonymous: Optional[bool] = None, version: Optional[str] = None, project: Optional[str] = None, log_model: Union[str, bool] = False, experiment=None, prefix: Optional[str] = "", agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None, agg_default_func: Optional[Callable[[Sequence[float]], float]] = None, **kwargs):
        super().__init__(name, save_dir, offline, id, anonymous, version, project, log_model, experiment, prefix, agg_key_funcs, agg_default_func, **kwargs)
        self.to_commit = {}
        self.to_commit_step = -1
    
    @rank_zero_only
    def commit(self, step=None):
        """the original log_metrics"""
        prev_step = self.to_commit_step
        if step > prev_step: 
            if len(self.to_commit) > 0:
                if step is not None:
                    self.to_commit["trainer/global_step"] = max(prev_step, 0)
                self.experiment.log(self.to_commit, step=max(0, prev_step))
                self.to_commit.clear()
            self.to_commit_step = step
        else:
            # continue to accumulate log
            pass
    
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """change to: cache to self.to_commit"""
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        self.to_commit.update(metrics)


def build_logger(cfg):
    if cfg.logging == 'none':
        from .logger import LFSLogger
        log = LFSLogger(project=cfg.project_name + osp.dirname(cfg.expname),
            name=osp.basename(cfg.expname),
            save_dir=cfg.exp_dir,
        )
    elif cfg.logging == 'wandb':
        os.makedirs(cfg.exp_dir + '/wandb', exist_ok=True)
        import wandb
        wandb.login(key=cfg.wandb_api)
        runid = None
        if os.path.exists(f"{cfg.exp_dir}/runid.txt"):
            runid = open(f"{cfg.exp_dir}/runid.txt").read()
        
        log = MyWandbLogger(
            project=cfg.project_name + osp.dirname(cfg.expname),
            name=osp.basename(cfg.expname),
            save_dir=osp.join(cfg.exp_dir, 'log'),
            id=runid,
            save_code=True,
            settings=wandb.Settings(start_method='thread'),
        )
        open(f"{cfg.exp_dir}/runid.txt", 'w').write(wandb.run.id)

    else:
        raise NotImplementedError
    return log