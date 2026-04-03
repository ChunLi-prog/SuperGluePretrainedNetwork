"""Job Settings."""
job_name = "default_jobname"
job_password = "newk8s666"
num_machines = 1
num_gpus_per_machine = 8
max_jobtime = 20160
framework = "pytorch"
task_label = "HDLT"
project_id = "Carizon_general_project"
output_bucket = "carizon_perception_2"
input_bucket = "carizon_perception, carizon_auto_eval, carizon_data_tos_bucket, carizon_perception_2"
work_dir_repfix = "/horizon-bucket/carizon_perception/....."

priority = 5
docker_image = (
    "cr-aidi-harbor-cn-shanghai-selfdriving-vecps.cr.autodriving.volcengine.com/imagesys/test:centos7.6-gcc5.4-py3.8-cuda11.1-maptr-2"
)


# launcher only for multi-machines
launcher = "mpi"

# upload folder
upload_folder_name = "k8s_job"
folder_list = [
    "./tools",
    "./projects",
    "./mmdetection3d",
    "./docs",
    "./checkpoints",
]
envs = {
    "HADOOP_HDFS_HOME": "${HADOOP_HOME}",
    # "HDFS_URL": "hdfs://hobot-bigdata",
    "PATH": "${WORKING_PATH}/gcc-5.1/bin:${PATH}",
    "OMP_NUM_THREADS": 1,
    "OPENBLAS_NUM_THREADS": 4,
}

job_list_single_node = [
    "python3 -m torch.distributed.run --nproc_per_node=${GPUS} --master_port=2333 \
    ${WORKING_PATH}/tools/train.py ${CONFIG} --cfg-option data_root=${DATA_ROOT} --launcher pytorch --deterministic --work-dir ${WORK_DIR}",
]
job_list_multi_node = [
    "NNODES=$(grep -n '' /job_data/hosts | wc -l)",
    "NODE_RANK=$(echo ${HOSTNAME: -1})",
    "MASTER_ADDR=$(head -n +1 /job_data/mpi_hosts)",
    "NGPUS=$(python3 -c 'import torch; print(torch.cuda.device_count())')",
    "MASTER_PORT=${MASTER_PORT:-29500}",

    "echo 'NNODES = ' ${NNODES} ', NODE_RANK = ' ${NODE_RANK} "
    "', NGPUS = ' ${NGPUS} ', MASTER_ADDR = ' ${MASTER_ADDR}",

    # Don't change those lines above this line.
    "TORCH_DISTRIBUTED_DEBUG=DETAIL python3 -m torch.distributed.run --nnodes=${NNODES} "
    "--node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} "
    "--nproc_per_node=${NGPUS} --master_port=${MASTER_PORT} "
    "${WORKING_PATH}/tools/train.py ${CONFIG} --cfg-option data_root=${DATA_ROOT} --launcher pytorch --deterministic --work-dir ${WORK_DIR}",
]

job_list = {
    "single_node": job_list_single_node,
    "multi_node": job_list_multi_node,
}

# custom_cmds = [
#     "ln -s /bucket/input/%s/data/pack_data ${WORKING_PATH}/tmp_data"
#     % input_bucket,
#     "ln -s /bucket/input/%s/models/bayes_release_models ${WORKING_PATH}/tmp_pretrained_models"  # noqa
#     % input_bucket,
#     "ln -s /job_data/models ${WORKING_PATH}/tmp_models",
# ]
custom_cmds = [
    "ln -s /cluster_home/custom_data/envs/maptr_1/maptr .",
    "ln -s /bucket/input/BasicAlgorithm/Users/yunchi.zhang/data_new .",
    "ln -s /bucket/input/BasicAlgorithm/Users/yunchi.zhang/nyc .",
]