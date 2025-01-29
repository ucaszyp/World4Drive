#!/usr/bin/env bash
CONFIG=$1
# echo $CONFIG
GPUS=$2
PORT=${PORT:-59230}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    train.py $CONFIG \
    --work-dir work_dirs/train_swin_wopre \
    --launcher pytorch ${@:3} \
    --deterministic \
    --cfg-options evaluation.jsonfile_prefix=work_dirs/$1/eval/results 
    # evaluation.classwise=True

# ./tools/nusc_my_train.sh law/default 8