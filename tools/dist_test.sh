#!/usr/bin/env bash
export NCCL_P2P_DISABLE=1
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
    test.py path/to/ckpt --launcher pytorch ${@:4} --eval bbox 