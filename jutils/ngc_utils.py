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
import subprocess
import re
import os
from argparse import ArgumentParser

# Template
# ngc batch run  \
# 	--preempt RUNONCE \
# 	--ace nv-us-west-2 \
#   --instance dgx1v.16g.1.norm \
# 	--image "nvidian/lpr/dexafford:conda_libgl" \
# 	--result /result \
# 	--org nvidian --team lpr  \
#   --workspace ws-judyye:ws-judyye:RW \
# 	--datasetid 102373:ho3d \
#   --total-runtime 3600s \
# 	--name "ml-model.res_dog" \
# 	--commandline ". /ws-judyye/conda/remote.sh; python -m jutils.ngc.res_dog" 


def add_ngc_args(arg_parser: ArgumentParser):
    arg_parser.add_argument("--slurm",action="store_true",)
    arg_parser.add_argument("--sl_dry",action="store_true",)
    arg_parser.add_argument("--sl_time",default=24, type=float, help='in hour')  # 16 hrs
    arg_parser.add_argument("--sl_ngpu",default=1, type=int)

    arg_parser.add_argument("--sl_org",default='nvidian', type=str)
    arg_parser.add_argument("--sl_team",default='lpr', type=str)
    arg_parser.add_argument("--sl_ace",default='nv-us-west-2', type=str)
    arg_parser.add_argument("--sl_preempt", default='RUNONCE')

    arg_parser.add_argument("--sl_ws",default='ws-judyye', type=str)
    arg_parser.add_argument("--sl_ws_src",default='/ws/', type=str)
    arg_parser.add_argument("--sl_data",default='102373:ho3d', type=str)

    arg_parser.add_argument("--sl_image",default='nvidian/lpr/dexafford:conda_libgl', type=str)
    arg_parser.add_argument("--sl_result", default='/result/', type=str)  

    return arg_parser


def ngc_wrapper(args, name, core_cmd):
    cmd = 'ngc batch run '
    cmd += ' --preempt %s' % args.sl_preempt
    cmd += ' --ace %s' % args.sl_ace
    cmd += ' --instance dgx1v.16g.%d.norm' % args.sl_ngpu
    cmd += ' --image %s' % args.sl_image
    cmd += ' --result %s ' % args.sl_result
    cmd += ' --org %s --team %s' % (args.sl_org, args.sl_team)
    cmd += ' --workspace %s:/%s:RW' % (args.sl_ws, args.sl_ws)
    for data in args.sl_data.split('+'):
        cmd += ' --datasetid %s' % args.sl_data
    cmd += ' --total-runtime %ds' % int(args.sl_time * 3600)
    cmd += ' --name ml-model.%s' % name
    
    print(core_cmd)
    ws_dst = os.path.join('/', args.sl_ws, os.getcwd().split(args.sl_ws_src)[1])
    setup_cmd = ' . /ws-judyye/conda/remote.sh; cd %s; ' % ws_dst
    cmd += r' --commandline "%s  %s"' % (setup_cmd, core_cmd)

    if not args.sl_dry:
        wrap_cmd(cmd)
    else:
        print(cmd)
    return


def ngc_engine():
    parser = ArgumentParser()
    add_ngc_args(parser)

    sl_args, unknown = parser.parse_known_args()

    print(unknown)
    # \$ -> \\\$
    unknown = [e.replace('$', '\\\\\\$') for e in unknown]  # because reading in will take 
    print(unknown)
    ngc_wrapper(
        sl_args, 
        re.sub(r'[^A-Za-z0-9\.]+', '', '.'.join(unknown)),  
        ' '.join(unknown))


class Worker():
    def __init__(self) -> None:
        pass

    def __call__(self, args):
        import importlib
        mod = importlib.import_module(args.worker)
        main_worker = getattr(mod, 'main_worker')
        main_worker(args)
        

def wrap_cmd(cmd):
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    try:
        print('wait')
        p.wait()
    except KeyboardInterrupt:
        try:
            print('Detect Ctrl+C, terminating....')
            p.terminate()
        except OSError:
            pass
        p.wait()



class Executor:
    def __init__(self, folder, local_str='', remote_str='') -> None:
        self.params = {}
        os.makedirs(folder, exist_ok=True)
        self.submit_dir = folder
        self.local_str = local_str
        self.remote_str = remote_str

    def update_parameters(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'total_runtime': 
                self.params['total-runtime'] = v                    
                print(v)
            else:
                self.params[k] = v
    
    def submit(self, func, func_args):
        import hydra.utils as hydra_utils
        cmd = self._param_to_str()
        MAIN_PID = os.getpid()
        fname = os.path.join(self.submit_dir, '%d_submit.pkl' % MAIN_PID)

        ws_dst = hydra_utils.get_original_cwd().replace(self.local_str, self.remote_str)

        import pickle
        with open(fname , 'wb') as fp:
            pickle.dump({'func': func, 'args': func_args}, fp) 

        # write batch file
        setup_cmd = ' . /ws-judyye/conda/remote.sh; cd %s; ' % ws_dst
        core_cmd = 'python -m jutils.ngc.exec_submit %s' % fname.replace(self.local_str, self.remote_str)
        cmd += ' --commandline "%s %s"' % (setup_cmd, core_cmd)
    
        os.makedirs(self.submit_dir, exist_ok=True)
        batch_file = self.submit_dir + '/%d_submit.sh' % MAIN_PID
        with open(batch_file, 'w') as wr_fp:
            wr_fp.write('echo HELLO\n')
            wr_fp.write('%s\n' % cmd)
            wr_fp.write('echo world\n')
        print('submitit! cache to %s' % batch_file)

        inner_cmd = '. %s' % batch_file
        wrap_cmd(inner_cmd)
    
    def _param_to_str(self, ):
        cmd = 'ngc batch run '        
        for k, v in self.params.items():
            print(k)
            if k == 'datasetid':
                for e in v.split('+'):
                    cmd += ' --datasetid %s ' % e
            else:
                cmd += ' --%s %s ' % (k, v)            
        
        return cmd


if __name__ == '__main__':
    ngc_engine()
