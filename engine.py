# --------------------------------------------------------
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# see CC-BY-NC-SA-4.0.md for details
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------

import os
import re
import hydra

from jutils.ngc_utils import Executor, Worker

@hydra.main(config_path='configs/', config_name='dev')
def main(args):
    # handle learning rate here! 
    # the original bs and lr are effective learningrate, adjust it according to gpu
    if hasattr(args, 'batch_size'):
        args.batch_size = args.batch_size // args.environment.ngpu
    if hasattr(args, 'learning_rate'):
        args.learning_rate = args.learning_rate * args.environment.ngpu
    
    if args.environment.slurm: 
        cfg = args.environment
        print(cfg)
        executor = Executor(
            os.path.join(cfg.local_dir, args.expname, 'submitit'), cfg.local_str, cfg.remote_str, cfg.cmd)
        name = re.sub(r'[^A-Za-z0-9\.]+', '_', args.expname)

        executor.update_parameters(
            name='ml-models.%s' % name,
            preempt=cfg.preempt,
            team=cfg.team,
            org=cfg.org,
            ace=cfg.ace,
            image=cfg.image,
            instance=cfg.instance,
            result=cfg.result,
            workspace='{0}:/{0}:RW'.format(cfg.workspace),
            total_runtime= '%ds' % int(cfg.time * 3600),
            datasetid=cfg.data,
        )
        
        executor.submit(Worker(), args)

    else:
        # main_worker(args)
        Worker()(args)
    


if __name__ == '__main__':
    main()
