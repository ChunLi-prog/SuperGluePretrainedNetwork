import os
import subprocess
import argparse
from mmengine.config import Config
# from cluster import CLUSTER

# cluster_dict = CLUSTER

def generate_job_yaml(cfg, upload_folder, run_cmd):
    yaml_name = "%s.yaml" % cfg.job_name
    with open(yaml_name, "w") as fn:
        fn.write("REQUIRED:\n")
        fn.write('  JOB_NAME: "%s"\n' % cfg.job_name)
        fn.write('  JOB_PASSWD: "%s"\n' % cfg.job_password)
        fn.write('  UPLOAD_DIR: "%s"\n' % os.path.basename(upload_folder))
        fn.write('  PROJECT_ID: "%s"\n' % cfg.project_id)
        fn.write("  WORKER_MIN_NUM: %d\n" % cfg.num_machines)
        fn.write("  WORKER_MAX_NUM: %d\n" % cfg.num_machines)
        fn.write("  GPU_PER_WORKER: %d\n" % cfg.num_gpus_per_machine)
        fn.write('  RUN_SCRIPTS: '+run_cmd+"\n")
        # fn.write('  RUN_SCRIPTS: "source /horizon-bucket/SD_Algorithm/08_perception_lidar/02_user/yuan.gao/minkunet.sh"\n')
        fn.write("OPTIONAL:\n")
        fn.write("  PRIORITY: %s\n" % cfg.priority)
        fn.write('  DOCKER_IMAGE: "%s"\n' % cfg.docker_image)
        fn.write("  WALL_TIME: %d\n" % cfg.max_jobtime)
        # set bucket
        if hasattr(cfg, "input_bucket"):
            fn.write("  DATA_SPACE:\n")
            fn.write('    DATA_TYPE: "dmp"\n')
            if type(cfg.input_bucket) == list:
                fn.write('    INPUT: "%s"\n' % (','.join(cfg.input_bucket)))
                # for i_bucket in cfg.input_bucket:
                #     fn.write('      - "%s"\n' % i_bucket)
            else:
                fn.write('    INPUT: "%s"\n' % cfg.input_bucket)
            if hasattr(cfg, "output_bucket") and cfg.output_bucket:
                fn.write('    OUTPUT: "%s"\n' % cfg.output_bucket)
    return yaml_name

def generate_upload_folder(cfg, upload_folder):
    assert "folder_list" in cfg
    for path in cfg.folder_list:
        if not os.path.exists(path):
            print(f"{path} not exists, skip")
            continue
        # support converting soft links to files.
        subprocess.check_call(["rsync", "-aL", path, upload_folder])
        print("copy %s to %s" % (path, upload_folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', type=str, default="carizon-share-4090")
    parser.add_argument('--sleep', action='store_true', default=False)
    parser.add_argument('--branch', type=str, default='baseline')
    parser.add_argument('--gpu', type=int, default=8)
    parser.add_argument('--task_label', type=str, default='minkunet')
    parser.add_argument('--job_name', default=None)
    args = parser.parse_args()

    cluster_name = args.cluster
    gpu_num = args.gpu
    if 'debug' in cluster_name:
        max_jobtime=59
        gpu_num=2
    else:
        max_jobtime=15000

    exp_name = "minkunet_"+args.branch
    # run_cmd_work = "echo ${WORKING_PATH} && cd ${WORKING_PATH} && bash ./run.sh "+args.branch+" "+str(gpu_num)
    run_cmd_work = "echo ${WORKING_PATH} && cd ${WORKING_PATH} && bash ./run.sh "+" "+",".join([str(i) for i in range(gpu_num)])
    run_cmd_sleep = "cd ${WORKING_PATH} && bash ./run_sleep.sh"
    run_cmd = run_cmd_sleep if args.sleep else run_cmd_work

    # docker pull hub.hobot.cc/imagesys/hat:pilot-torch1.9.1-cuda11.1-20211101
    k8s_config = dict(
        job_name="lidarseg_"+exp_name,
        task_label=args.task_label,
        job_password="1010",
        num_machines=1,
        num_gpus_per_machine=gpu_num,
        framework="pytorch",
        project_id="Carizon_general_project",
        input_bucket="carizon_orig, carizon_perception",
        priority=5,
        docker_image = "cr-aidi-harbor-cn-shanghai-selfdriving-vecps.cr.autodriving.volcengine.com/imagesys/hat:fsd_multitask-cu11-20230616-v3.9",
        # default 7200 = 5days
        max_jobtime=max_jobtime,
        # launcher only for multi-machines
        # launcher="mpi",
        # upload folder
        upload_folder_name="mmdetection3d-horizon",
        folder_list=[
            "../mmdetection3d-horizon",
            "./run.sh",
            "./run_sleep.sh"
        ],
        # cluster='share-debug-queue-bcloud'
        cluster=cluster_name,
        # job_list=job_list,
    )
    if args.job_name is not None:
        k8s_config["job_name"] = args.job_name

    print("submit.py")
    print("================================")
    print("BRANCH_NAME="+args.branch)
    print("GPU_NUM="+str(gpu_num))
    print("CLUSTER="+k8s_config["cluster"])
    print("RUN="+run_cmd)
    print("================================")

    print("submit config:")
    for k in k8s_config:
        print(k, "=", k8s_config[k])
    print("================================")

    cfg = Config(k8s_config)
    subprocess.check_call(["mkdir", "-p", cfg.upload_folder_name])
    yaml_name = generate_job_yaml(cfg, cfg.upload_folder_name, run_cmd)
    assert os.path.exists(yaml_name), "Cannot generate yaml successfully."

    generate_upload_folder(cfg, cfg.upload_folder_name)
    # import ipdb
    # ipdb.set_trace()

    # generate_bash_file(cfg, args.upload_folder, args.sleep)
    # traincli submit -f waymo_and_nuscenes_3d_challenge.yaml --current-cluster share-2080ti-ucloud
    # cmd = ["traincli", "submit", "-f", yaml_name, "--current-cluster", cfg.cluster]
    cmd = ["aidi-inf-cli", "job", "submit", "-f", yaml_name, "--current-cluster", cfg.cluster]
    print(k8s_config)
    print(cmd)

    try:
        subprocess.check_call(cmd)
        print("Submit job successfully.")
    except Exception as e:
        print(e)
        raise Exception("error")
    finally:
        if os.path.exists(cfg.upload_folder_name):
            subprocess.check_call(["rm", "-rf", cfg.upload_folder_name])
