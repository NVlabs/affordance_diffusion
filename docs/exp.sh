python -m models.base -m  --config-name=test  \
  data.data_dir='/ws-judyye/data/epic/obj_w_kpt/*.*g'  test_name=hijack test_num=1  \
  mode=hijack \
  environment.slurm=False 2>&1 | tee output.log 


python -m models.base -m  --config-name=train \
  expname=reproduce/\${model.module} \
  model=content_ldm  \
  environment.slurm=False logging=none 2>&1 | tee output_train.log

python -m models.base -m  --config-name=train \
  expname=reproduce/\${model.module} \
  model=content_glide \
  environment.slurm=False logging=none 2>&1 | tee output_train.log


python -m models.base -m  --config-name=train \
  expname=reproduce/\${model.module} \
  'data@trainsets=[hoi4d_train]' 'data@testsets=[hoi4d_test]' \
  model=layout environment.mem=32 \
  environment.slurm=False logging=none 2>&1 | tee output_train.log



python inference.py --config-name=test  \
    environment.slurm=False 2>&1 | tee output.log


python inference.py --config-name=test  \
  test_name=tmp_ldm \
  what_ckpt=/ws-judyye/result/release/content_ldm/checkpoints/last.ckpt \
  data.data_dir='/ws-judyye/data/visor/more_obj/*.*g' \
    environment.slurm=False 2>&1 | tee output.log


python inference.py  \
  test_name=tmp \
  what_ckpt=/ws-judyye/result/release/content_ldm/checkpoints/last.ckpt \
  data.data_dir='/ws-judyye/data/visor/more_obj/*.*g' \

python -m scripts.interpolate \
  dir=/ws-judyye/result/release/layout/cascade 