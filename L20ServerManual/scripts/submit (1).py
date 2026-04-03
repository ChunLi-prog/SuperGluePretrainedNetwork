# Copyright (c) Horizon Robotics. All rights reserved.
# submit jobs

import argparse
import os
import subprocess

from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--job", "-j", type=str, required=True)
    parser.add_argument("--cluster-debug", action="store_true")
    parser.add_argument("--k8config", "-k", type=str, default="submit/k8s_config.py")
    parser.add_argument("--upload-folder", type=str, default="./")
    parser.add_argument(
        "--save-upload-folder",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--current-cluster",
        default=None,
        type=str,
        help="[traincli option] If None, use default in yaml",
    )
    parser.add_argument(
        "--debug", dest="debug", action="store_true", help="[traincli option]"
    )
    parser.add_argument(
        "--sleep",
        dest="sleep",
        action="store_true",
        help="sleep in cluster for debug",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/tmp"
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_gpus_per_machine",
        type=int,
        default=8,
    )
    return parser.parse_args()


def generate_job_yaml(cfg, upload_folder):
    yaml_name = os.path.join(upload_folder, "%s.yaml" % cfg.job_name)
    with open(yaml_name, "w") as fn:
        fn.write("REQUIRED:\n")
        fn.write('  JOB_NAME: "%s"\n' % cfg.job_name)
        fn.write('  JOB_PASSWD: "%s"\n' % cfg.job_password)
        fn.write('  UPLOAD_DIR: "%s"\n' % os.path.basename(upload_folder))
        fn.write('  PROJECT_ID: "%s"\n' % cfg.project_id)
        fn.write("  WORKER_MIN_NUM: %d\n" % cfg.num_machines)
        fn.write("  WORKER_MAX_NUM: %d\n" % cfg.num_machines)
        fn.write("  GPU_PER_WORKER: %d\n" % cfg.num_gpus_per_machine)
        fn.write('  RUN_SCRIPTS: "${WORKING_PATH}/job.sh"\n')
        fn.write("OPTIONAL:\n")
        fn.write("  PRIORITY: %s\n" % cfg.priority)
        fn.write('  DOCKER_IMAGE: "%s"\n' % cfg.docker_image)
        fn.write("  WALL_TIME: %d\n" % cfg.max_jobtime)
        # set bucket
        if hasattr(cfg, "input_bucket"):
            fn.write("  DATA_SPACE:\n")
            fn.write('    DATA_TYPE: "dmp"\n')
            fn.write('    INPUT: "%s"\n' % cfg.input_bucket)
            if hasattr(cfg, "output_bucket") and cfg.output_bucket:
                fn.write('    OUTPUT: "%s"\n' % cfg.output_bucket)
        elif hasattr(cfg, "output_bucket") and cfg.output_bucket:
            fn.write("  DATA_SPACE:\n")
            fn.write('    DATA_TYPE: "dmp"\n')
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


def write_multi_machines_cmd(fn, cfg, job):
    """Write job to fn."""
    command = "mpirun -n %d -ppn %d --hostfile %s %s" % (
        cfg.num_machines,
        1,
        "/job_data/mpi_hosts",
        job
    )
    fn.write("%s\n" % command)


def generate_bash_file(cfg, upload_folder, run_in_sleep):
    expname = cfg.config.split('/')[-1].split('.py')[0]

    bash_file = os.path.join(upload_folder, "job.sh")
    with open(bash_file, "w") as fn:
        fn.write("set -e\n")
        fn.write("export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH\n")
        fn.write("date\n")
        fn.write("env\n")
        fn.write("pip3 list\n")
        fn.write("export PYTHONUNBUFFERED=0\n")
        fn.write(f"export GPUS={cfg.num_gpus_per_machine}\n")
        fn.write(f"export CONFIG={cfg.config}\n")
        fn.write(f"export WORK_DIR={cfg.config}\n")
        fn.write("export WORK_DIR={}\n".format(os.path.join(cfg.work_dir_repfix, expname)))
        fn.write("export LOGDIR={}\n".format(os.path.join(cfg.work_dir_repfix, expname, "tf_logs")))
        fn.write("mkdir -p ${WORK_DIR}\n")
        fn.write("ln -s /job_tboard/ ${LOGDIR}\n")
        for env_name in cfg.get("envs", {}):
            env_value = cfg.envs[env_name]
            fn.write(f"export {env_name}={env_value}\n")
        fn.write("cd ${WORKING_PATH}\n")
        if run_in_sleep:
            fn.write("sleep 1000000m\n")

        for cmd in getattr(cfg, "prefix_cmds_on_master", []):
            fn.write(f"{cmd}\n")

        if hasattr(cfg, "custom_cmds"):
            assert not hasattr(cfg, "custom_cmds_before_job_list")
            cfg.custom_cmds_before_job_list = cfg.custom_cmds

        if cfg.num_machines > 1:  # multi-machines
            dir_name = os.path.dirname(__file__)
            url2ip_path = os.path.join(dir_name, "./url2IP.py")
            subprocess.check_call(["cp", url2ip_path, upload_folder])
            fn.write("python3 url2IP.py\n")
            fn.write("cat /job_data/mpi_hosts\n")
            fn.write("dis_url=$(head -n +1 /job_data/mpi_hosts)\n")
            fn.write("unset HOSTNAME\n")
            if cfg.launcher == "mpi":
                if hasattr(cfg, "custom_cmds_before_job_list"):
                    cmds_file = os.path.join(
                        upload_folder, "custom_cmds_before_job_list.sh"
                    )
                    with open(cmds_file, "w") as cus:
                        for cmd in cfg.custom_cmds_before_job_list:
                            cus.write("%s\n" % cmd)
                    job = "bash ${WORKING_PATH}/custom_cmds_before_job_list.sh"
                    write_multi_machines_cmd(fn, cfg, job)

                if hasattr(cfg, "job_list") and cfg.job_list:
                    cmds_file = os.path.join(upload_folder, "job_list.sh")
                    with open(cmds_file, "w") as jobs:
                        for cmd in cfg.job_list["multi_node"]:
                            jobs.write("%s\n" % cmd)
                    job = "bash ${WORKING_PATH}/job_list.sh"
                    write_multi_machines_cmd(fn, cfg, job)

                if hasattr(cfg, "custom_cmds_after_job_list"):
                    cmds_file = os.path.join(
                        upload_folder, "custom_cmds_after_job_list.sh"
                    )
                    with open(cmds_file, "w") as cus:
                        for cmd in cfg.custom_cmds_after_job_list:
                            cus.write("%s\n" % cmd)
                    job = "bash ${WORKING_PATH}/custom_cmds_after_job_list.sh"
                    write_multi_machines_cmd(fn, cfg, job)
        else:
            if hasattr(cfg, "custom_cmds_before_job_list"):
                for cmd in cfg.custom_cmds_before_job_list:
                    fn.write("%s\n" % cmd)
            for job in cfg.job_list["single_node"]:
                fn.write("%s\n" % job)
            if hasattr(cfg, "custom_cmds_after_job_list"):
                for cmd in cfg.custom_cmds_after_job_list:
                    fn.write("%s\n" % cmd)

        for cmd in getattr(cfg, "suffix_cmds_on_master", []):
            fn.write(f"{cmd}\n")

    subprocess.check_call(["chmod", "777", bash_file])


def refine_cfg(cfg, args):
    cfg.job_name = args.job
    cfg.config = args.config
    cfg.num_machines = args.num_machines
    cfg.num_gpus_per_machine = args.num_gpus_per_machine

    if args.cluster_debug:
        cfg.num_machines, cfg.num_gpus_per_machine, cfg.max_jobtime = 1, 2, 60
        cfg.job_name = "debug_" + args.job

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.k8config)
    if "k8s_config" in cfg:
        cfg = Config(cfg["k8s_config"])
    refine_cfg(cfg, args)
    args.upload_folder = os.path.join(
        args.upload_folder, cfg.upload_folder_name
    )
    subprocess.check_call(["mkdir", "-p", args.upload_folder])

    if "on_generate_job_yaml_begin" in cfg:
        assert callable(cfg.on_generate_job_yaml_begin)
        cfg.on_generate_job_yaml_begin(cfg, args)

    yaml_name = generate_job_yaml(cfg, args.upload_folder)
    assert os.path.exists(yaml_name), "Cannot generate yaml successfully."

    generate_upload_folder(cfg, args.upload_folder)

    generate_bash_file(cfg, args.upload_folder, args.sleep)

    cmd = ["aidi-inf-cli", "job","submit", "-f", yaml_name, "-t", args.log_dir]
    if args.current_cluster:
        cmd += ["--queue_name", args.current_cluster]
    if args.debug:
        cmd += ["--debug"]

    try:
        subprocess.check_call(cmd)
        print("Submit job successfully.")
    except Exception as e:
        print(e)
        raise Exception("error")
    finally:
        if not args.save_upload_folder:
            if os.path.exists(args.upload_folder):
                subprocess.check_call(["rm", "-rf", args.upload_folder])
