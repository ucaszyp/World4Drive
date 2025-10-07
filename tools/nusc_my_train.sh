#!/usr/bin/env bash
CONFIG=$1
# echo $CONFIG
GPUS=$2
PORT=${PORT:-59550}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    train.py $CONFIG \
    --work-dir work_dirs/w4d_24_dino_with_grad_and_metric_depth \
    --launcher pytorch ${@:3} \
    --deterministic \
    --cfg-options evaluation.jsonfile_prefix=work_dirs/$1/eval/results \
    --resume-from /data/zyp/World4Drive_v1/work_dirs/w4d_24_dino_with_grad_and_metric_depth/epoch_1.pth
    # evaluation.classwise=True