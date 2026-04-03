#!/bin/bash

cd /running_package/
virtualenv -p //usr/local/bin/python3 env_3
source env_3/bin/activate
cd mmdetection3d-horizon/mmdetection3d-horizon
pip3 install -r requirements.txt -i https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simple
pip3 install ./mmcv-2.0.0rc4-cp38-cp38-manylinux1_x86_64.whl



# 判断传入的参数数量
#if [ "$#" -lt 2 ]; then
    #echo "Usage: $0 GPUID PORT"
        #exit 1
        #fi
# 获取 GPU ID
GPUID=$1
# 判断 GPUID 中的 ',' 数量，从而确定 GPU 的数量
num_commas=$(echo $GPUID | tr -cd ',' | wc -c)
#num_gpu 为逗号数量 + 1
num_gpu=$((num_commas + 1))
#num_gpu=$1
#num_gpu=2
# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPUID
#export PORT=$2
# 获取当前 git 分支名称
current_branch=$(git rev-parse --abbrev-ref HEAD)
# 使用当前分支名称
echo "You are currently on branch: $current_branch"
# 设置工作路径
config_file="configs/minkunet/lidar-seg_minkunet_contrast.py"
if [ "$DEBUG" == "1" ]; then
    export PYTHONBREAKPOINT=
    #work_dir="--work-dir=./test_dir/$current_branch"
    work_dir="--work-dir=/job_data"
    echo "**************DEBUG MODE***************"
    config_file="configs/minkunet/lidar-seg_minkunet_contrast_debug.py"
else
    export PYTHONBREAKPOINT=1
    #work_dir="--work-dir=./test_dir/$current_branch"
    work_dir="--work-dir=/job_data"
fi
# 运行脚本
echo "Using config file: $config_file"
bash tools/dist_train.sh $config_file $num_gpu $work_dir
