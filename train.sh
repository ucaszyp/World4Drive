export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./tools/nusc_my_train.sh ./projects/configs/w4d/default.py 8