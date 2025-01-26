# Overview
This is the upgrade version of LAW (Enhancing End-to-End Autonomous Driving with Latent World Model). We removed the view selection component and revised several modules to achieve a more universal framework.


# Step-by-step installation instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n law python=3.8 -y
conda activate law
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install -r requirements.txt
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-5 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0

```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install timm.**
```shell
pip install timm
```

**f. Install mmdet3d.**
```shell
conda activate law
git clone https://github.com/open-mmlab/mmdetection3d.git
cd /path/to/mmdetection3d
git checkout -f v0.17.1
python setup.py develop
```

**g. Install nuscenes-devkit.**
```shell
pip install nuscenes-devkit==1.1.9
pip install yapf==0.40.1
```

**h. Follow VAD to organize nuscenes and download the pickle**
```shell
$WORK_DIR/data/nuscenes/
```
For details, please refer to docs/prepare_dataset.md

# Training
```shell
  ./tools/nusc_my_train.sh law/default 8
```

# Testing
```shell
  ./tools/dist_test $CONFIG $CKPT $NUM_GPU
```