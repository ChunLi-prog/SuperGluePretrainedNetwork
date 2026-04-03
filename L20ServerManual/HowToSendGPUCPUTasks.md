帐号准备
1. aidi帐号
2. aidi集群权限/appid等配置
3. aidi本地配置和gpuclutser配置
  1. .aidisdk/config.yaml
以上参阅新人指南, 不加详述
提交集群
以lidar分割网络为例
https://gitlab.carizon.work/algorithm/maploc/cloudmodel/waymoseg/-/tree/feature-lidar_seg_minkunet_hde_jv
代码结构如下
[图片]
其中mmdetection3d-horizon可以替换成其他开源代码
重点是submit文件夹
Submit
submit.py
run.sh
run_sleep.sh
submit.py
先从submit.py讲起, 功能分别是
1. 打包代码包
2. 上传代码包
3. 删掉临时代码包
4. 在云端调用run.sh或者run_sleep.sh
正常情况下只用这个就可以
 重点看k8s_config部分
    k8s_config = dict(
        job_name="lidarseg_"+exp_name, # 实验名字, 随便配
        task_label=args.task_label, # 实验名字, 随便配
        job_password="1010",  # 用默认就行, 不用管
        num_machines=1,  # 机器数量, 
        num_gpus_per_machine=gpu_num,  # 每台机器几张卡, 这里配成8就是1机8卡
        
        # 举例: 需要16卡(2机8卡), 就把num_machines配成2, num_gpus_per_machine配成8
       
        framework="pytorch",
        project_id="Carizon_general_project",
        input_bucket="carizon_orig, carizon_perception",
        priority=5,
        # 根据cuda版本选docker
        docker_image = "cr-aidi-harbor-cn-shanghai-selfdriving-vecps.cr.autodriving.volcengine.com/imagesys/hat:fsd_multitask-cu11-20230616-v3.9",
        # 根据任务选择最大时间, 超过时间会被自动kill
        # default 7200 = 5days
        max_jobtime=max_jobtime,
        # 上传的文件夹名, 自己配置
        upload_folder_name="mmdetection3d-horizon",
        # 拷贝到上面文件夹的文件, 自己配置
        folder_list=[
            "../mmdetection3d-horizon",
            "./run.sh",
            "./run_sleep.sh"
        ],
        # 集群名
        cluster=cluster_name,
    )
运行submit.py
# 正常运行
python submit.py --cluster carizon-4dgt-gpu --gpu 8
# sleep 运行
python submit.py --cluster carizon-4dgt-gpu --gpu 8 --sleep
run.sh
作用: 
- 创建虚拟环境
- 安装依赖
- 调用代码包中代码, 多卡训练
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
如需手动调用, 参考:
./run.sh 0,1,2,3,4,5,6,7 # 8卡
run_sleep.sh
Debug用, 会占住集群资源不释放, 用于在集群云端调试
sleep 20000m
在集群测试
在一开始, 并不清楚自己改的submit.py / run.sh 或者自己的代码能否跑通, 需要在集群云端测试
1. 提交一个sleep任务
# sleep 运行
python submit.py --cluster carizon-4dgt-gpu --gpu 2 --sleep 
2. 网页端进入集群任务
[图片]
[图片]
[图片]
3. 集群调试
[图片]
进入集群调试页面, 可自行调试, 建议开启一个tmux, 在网页断连后依然可以用tmux attach恢复
[图片]
上传的代码包在running_package下, job_data和job_log 分别对应日志和输出两个页面,  
[图片]
建议将模型保存在 job_data, 日志保存在 job_log下
4. 调试结束后, 自行终结训练任务, 防止集群空转
多机多卡手动开启指南
假如1机8卡不够用, 需要用到2机8卡或者更多机器, 可以参考以下设置
1. 确认下代码中是否有torch.distributed的调用方式, 例如mmdet3d中
[图片]
或者Pointcept中:
[图片]
2. 修改submit.py中的k8s_config
  1. 例如需要16卡(2机8卡), 就把num_machines配成2, num_gpus_per_machine配成8
  2. 通过submit.py --sleep 方式提交job
3. 查看pod的ip
  1. 例如2机8卡, aidi中会存在两个pod, 其中后缀0是主机
  2. 进去pod0, 通过ifconfig命令, 查看主机ip
[图片]
[图片]
4. 分别在各个pod上, 根据脚本描述设置参数
  1. nnodes/num-machines 表示机器个数, 例如2机设为2
  2. node_rank/machine-rank 表示主次关系, 例如主机设为0, 其余机器依次1/2/3等
  3. master_addr/dist-url  表示主机ip, 例如 tcp://192.168.67.28
  4. master_port 表示主机的port, 这个可以随意设置, 但要保持一致
  5. nproc_per_node/num_gpu 表示每台机器有几个gpu, 例如8卡
5. 分别在各pod上, 启动训练脚本即可
  1. 如果挂了, 或者提示port被占用, 就统一设置一个新的port, 重新启动即可
  
多机多卡适配脚本
【2024/11/18 更新】
Submit
- 新建submit文件夹，可以放在项目目录内。
- submit文件夹包含3个py文件：
  - k8s_config.py
    - 包含绝大部分配置项，如输入/输出bucket，上传folder等，需要根据个人实际需求自行修改
暂时无法在飞书文档外展示此内容
  - submit.py
    - 提交集群运行脚本，一般不用修改
暂时无法在飞书文档外展示此内容
  - url2IP.py
    - 无需修改
暂时无法在飞书文档外展示此内容
Tips
开发机和集群只能连地平线的pypi资源, 即
pip3 install -r requirements.txt -i https://pypi.hobot.cc/simple --extra-index-url https://pypi.hobot.cc/hobot-local/simpleshell
没有的话需要自行上传whl或从源码安装
