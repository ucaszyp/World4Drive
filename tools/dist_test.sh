#!/usr/bin/env bash

PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
#     test.py projects/configs/law/default.py work_dirs/train1/epoch_10.pth --launcher pytorch ${@:4} --eval bbox

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
#     test.py projects/configs/law/default.py work_dirs/train1/epoch_11.pth --launcher pytorch ${@:4} --eval bbox

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
#     test.py projects/configs/law/default.py work_dirs/train1/epoch_12.pth --launcher pytorch ${@:4} --eval bbox

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
#     test.py projects/configs/law/default.py work_dirs/train1/epoch_13.pth --launcher pytorch ${@:4} --eval bbox

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
#     test.py projects/configs/law/default.py work_dirs/train1/epoch_14.pth --launcher pytorch ${@:4} --eval bbox

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
#     test.py projects/configs/law/default.py work_dirs/train1/epoch_15.pth --launcher pytorch ${@:4} --eval bbox

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
#     test.py projects/configs/law/default.py work_dirs/train1/epoch_16.pth --launcher pytorch ${@:4} --eval bbox

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
#     test.py projects/configs/law/default.py work_dirs/train1/epoch_17.pth --launcher pytorch ${@:4} --eval bbox

python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT \
    test.py projects/configs/law/default.py work_dirs/train1/epoch_23.pth --launcher pytorch ${@:4} --eval bbox